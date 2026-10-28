"""Shared deployment availability and committed-base eligibility resolution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
import time

from chartreux.core.config.models import ModelConfig, ThinkingLevel
from chartreux.core.llm.backend.anthropic import REASONING_BLOCK_TYPES
from chartreux.core.llm.failures import FailureCategory, FailureInfo
from chartreux.core.llm.thinking_levels import get_thinking_levels
from chartreux.core.llm_models import LLMMessage
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.resolver import ModelResolver, ResolvedModel
from chartreux.core.recovery_models import canonical_endpoint_identity
from chartreux.core.recovery_override import CooldownOverrideLedger, OverridePermit
from chartreux.core.session_types import CommittedModelIdentity


class ExclusionReason(StrEnum):
    COOLDOWN = "cooldown"
    ENDPOINT_COOLDOWN = "endpoint-cooldown"
    IMAGES_UNSUPPORTED = "images-unsupported"
    THINKING_UNSUPPORTED = "thinking-unsupported"
    REASONING_UNSUPPORTED = "reasoning-unsupported"
    COMPACTION_INCOMPATIBLE = "compaction-incompatible"
    DISABLED = "disabled"
    ALLOWLIST_EXCLUDED = "allowlist-excluded"


@dataclass(frozen=True, slots=True)
class DeploymentExclusion:
    base_model: str
    provider: str
    wire_name: str
    reason: ExclusionReason
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class EligibleDeployment:
    resolved: ResolvedModel
    recovery_probe: bool = False
    recovery_probe_token: int | None = None


@dataclass(frozen=True, slots=True)
class EligibilityResult:
    candidates: tuple[EligibleDeployment, ...]
    exclusions: tuple[DeploymentExclusion, ...]


class AllDeploymentsUnavailableError(ValueError):
    """No deployment of the committed base can serve the current completion."""

    code = "all_deployments_unavailable"

    def __init__(
        self, base_model: str, exclusions: Sequence[DeploymentExclusion]
    ) -> None:
        super().__init__(
            f"All deployments for base model {base_model!r} are unavailable"
        )
        self.base_model = base_model
        self.exclusions = tuple(exclusions)


@dataclass(slots=True)
class _Cooldown:
    until: float
    probe_in_flight: bool = False
    probe_token: int | None = None
    #: Wall-clock time the probe claim was taken (round-2 V3); None when no
    #: claim is outstanding.
    claimed_at: float | None = None


@dataclass(frozen=True, slots=True)
class _EndpointFailureEvent:
    """One observed failure on one canonical endpoint identity (WP1)."""

    observed_at: float
    base_model: str
    category: FailureCategory | None
    signature: str | None


#: Bound on retained endpoint events per identity (oldest are dropped first).
_MAX_ENDPOINT_EVENTS = 128

#: Default wall-clock TTL of a single-flight recovery probe claim (round-2
#: V3). A claim older than this is treated as dead so a stranded claim never
#: probe-blocks a deployment past this bound. The bound is sized for the
#: DEFAULT configuration only: DEFAULT_API_TIMEOUT 720s +
#: DEFAULT_API_RETRY_MAX_ELAPSED_TIME 300s = 1020s < 1200s. The httpx read
#: timeout is an idle per-read timeout, not a total deadline
#: (backend/generic.py: httpx.Timeout(timeout, connect=..., write=...,
#: pool=...)), so under non-default config (api_timeout +
#: api_retry_max_elapsed_time > 1200s) or a slow-drip stream a live claim
#: CAN outlive this TTL — the bounded consequence is one extra concurrent
#: probe; settlement stays token-owned.
_DEFAULT_PROBE_CLAIM_TTL = 1200.0


class AvailabilityRegistry:
    """Root-owned mutable cooldown state shared by every runtime child."""

    def __init__(
        self,
        *,
        initial_cooldown: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        endpoint_correlation_window: float = 300.0,
        endpoint_correlation_min_events: int = 2,
        endpoint_correlation_min_models: int = 2,
        endpoint_cooldown: float | None = None,
        probe_claim_ttl: float = _DEFAULT_PROBE_CLAIM_TTL,
    ) -> None:
        if initial_cooldown < 0:
            raise ValueError("initial_cooldown must be non-negative")
        if endpoint_correlation_window <= 0:
            raise ValueError("endpoint_correlation_window must be positive")
        if endpoint_correlation_min_events < 1:
            raise ValueError("endpoint_correlation_min_events must be >= 1")
        if endpoint_correlation_min_models < 1:
            raise ValueError("endpoint_correlation_min_models must be >= 1")
        if endpoint_cooldown is not None and endpoint_cooldown < 0:
            raise ValueError("endpoint_cooldown must be non-negative")
        if probe_claim_ttl <= 0:
            raise ValueError("probe_claim_ttl must be positive")
        self.initial_cooldown = initial_cooldown
        self.probe_claim_ttl = probe_claim_ttl
        self._clock = clock
        self._cooldowns: dict[tuple[str, str], _Cooldown] = {}
        self._next_probe_token = 0
        self.endpoint_correlation_window = endpoint_correlation_window
        self.endpoint_correlation_min_events = endpoint_correlation_min_events
        self.endpoint_correlation_min_models = endpoint_correlation_min_models
        self.endpoint_cooldown = (
            initial_cooldown if endpoint_cooldown is None else endpoint_cooldown
        )
        self._endpoint_events: dict[str, list[_EndpointFailureEvent]] = {}
        self._endpoint_cooldowns: dict[str, float] = {}

    def record_failure(
        self,
        base_model: str,
        provider: str,
        failure: FailureInfo | None = None,
        *,
        endpoint: str | None = None,
        signature: str | None = None,
        probe_token: int | None = None,
    ) -> None:
        """Start or extend cooldown and release an outstanding recovery probe.

        Endpoint correlation (WP1) is recorded separately from the
        ``(base_model, provider)`` cooldown: repeated rate-limit failures
        with an identical structured signature across enough distinct base
        models on ONE endpoint arm an endpoint-wide cooldown. A single
        isolated timeout never broadens to endpoint-wide. ``endpoint`` is a
        raw endpoint URL; it is canonicalized (userinfo stripped) before it
        keys the correlation state, so the key space can never split.

        Review R13: probe SETTLEMENT is token-owned like probe release — a
        failure that does not carry the in-flight probe's owner token (a
        late normal attempt) extends the cooldown but never replaces the
        probe claim; only the owner's own failure settles (and releases)
        it.
        """
        delay = max(
            self.initial_cooldown,
            failure.retry_after if failure is not None and failure.retry_after else 0.0,
        )
        key = (base_model, provider)
        until = self._clock() + delay
        current = self._cooldowns.get(key)
        if (
            current is not None
            and current.probe_in_flight
            and current.probe_token != probe_token
        ):
            # A newer in-flight recovery probe owns the single-flight
            # claim: the failure still extends the cooldown, but the probe
            # state survives the settlement.
            current.until = max(until, current.until)
        else:
            self._cooldowns[key] = _Cooldown(
                until=max(until, current.until if current is not None else until)
            )
        canonical_endpoint = canonical_endpoint_identity(endpoint)
        if canonical_endpoint is not None:
            self._record_endpoint_failure(
                canonical_endpoint, base_model, failure, signature
            )

    def _record_endpoint_failure(
        self,
        endpoint: str,
        base_model: str,
        failure: FailureInfo | None,
        signature: str | None,
    ) -> None:
        """Retain one bounded endpoint event and try to arm correlation."""
        now = self._clock()
        events = self._endpoint_events.setdefault(endpoint, [])
        cutoff = now - self.endpoint_correlation_window
        while events and events[0].observed_at < cutoff:
            events.pop(0)
        if len(events) >= _MAX_ENDPOINT_EVENTS:
            events.pop(0)
        events.append(
            _EndpointFailureEvent(
                observed_at=now,
                base_model=base_model,
                category=failure.category if failure is not None else None,
                signature=signature,
            )
        )
        self._arm_endpoint_correlation(endpoint, now)

    def _arm_endpoint_correlation(self, endpoint: str, now: float) -> None:
        """Arm an endpoint-wide cooldown from cross-model rate-limit evidence.

        Requires, within the correlation window: at least
        ``endpoint_correlation_min_events`` rate-limit failures sharing one
        identical NON-EMPTY structured signature, observed across at least
        ``endpoint_correlation_min_models`` distinct base models. Anything
        less is noise, not an endpoint signal.
        """
        cutoff = now - self.endpoint_correlation_window
        events = [
            event
            for event in self._endpoint_events.get(endpoint, ())
            if event.observed_at >= cutoff
            and event.category is FailureCategory.RATE_LIMIT
            and event.signature
        ]
        if len(events) < self.endpoint_correlation_min_events:
            return
        models_by_signature: dict[str, set[str]] = {}
        for event in events:
            assert event.signature is not None
            models_by_signature.setdefault(event.signature, set()).add(event.base_model)
        if not any(
            len(models) >= self.endpoint_correlation_min_models
            for models in models_by_signature.values()
        ):
            return
        until = now + self.endpoint_cooldown
        current = self._endpoint_cooldowns.get(endpoint)
        self._endpoint_cooldowns[endpoint] = max(
            until, current if current is not None else until
        )

    def endpoint_correlation_active(self, endpoint: str | None) -> bool:
        """Whether an endpoint-wide correlation cooldown is currently armed."""
        return self.endpoint_cooldown_until(endpoint) is not None

    def endpoint_cooldown_until(self, endpoint: str | None) -> float | None:
        """Expiry of the endpoint-wide correlation cooldown; None = not armed."""
        if not endpoint:
            return None
        until = self._endpoint_cooldowns.get(endpoint)
        if until is None:
            return None
        if self._clock() >= until:
            del self._endpoint_cooldowns[endpoint]
            return None
        return until

    def record_success(
        self, base_model: str, provider: str, *, probe_token: int | None = None
    ) -> None:
        """Clear cooldown after a successful normal attempt or recovery probe.

        Endpoint correlation cooldowns are deliberately NOT cleared: they are
        time-based endpoint evidence, and any single success must not
        broaden-clear them (the fleet-wide-clear defect class).

        Review R13: probe SETTLEMENT is token-owned like probe release — a
        success that does not carry the in-flight probe's owner token (a
        late normal attempt) lifts the cooldown but never deletes the probe
        claim; only the owner's own success settles it.
        """
        state = self._cooldowns.get((base_model, provider))
        if (
            state is not None
            and state.probe_in_flight
            and state.probe_token != probe_token
        ):
            # A newer in-flight recovery probe owns the single-flight
            # claim: the success lifts the cooldown (a success never
            # extends it), but the probe state survives the settlement.
            state.until = min(state.until, self._clock())
            return
        self._cooldowns.pop((base_model, provider), None)

    def reset_base(self, base_model: str) -> None:
        """Clear every deployment cooldown for a user-requested base recovery.

        Endpoint correlation cooldowns are deliberately NOT cleared here
        either; they expire on their own clock.
        """
        for key in [key for key in self._cooldowns if key[0] == base_model]:
            del self._cooldowns[key]

    def _probe_claim_stale(self, state: _Cooldown) -> bool:
        """Whether a claimed probe's wall-clock TTL has expired (round-2 V3).

        A claim is dated when it is taken; a claim older than
        ``probe_claim_ttl`` is treated as dead — it stops blocking
        availability and can be re-claimed. The TTL is a heuristic bound,
        not a guarantee: it is sized for the default request timeout plus
        retry budget (720s + 300s), and the httpx read timeout is an idle
        per-read timeout rather than a total deadline, so a live claim under
        non-default config or a slow-drip stream can outlive it (bounded
        consequence: one extra concurrent probe; settlement stays
        token-owned). Read-only: staleness never mutates the claim itself.
        """
        return (
            state.probe_in_flight
            and state.claimed_at is not None
            and self._clock() - state.claimed_at >= self.probe_claim_ttl
        )

    def is_available(
        self, base_model: str, provider: str, *, endpoint: str | None = None
    ) -> bool:
        """Check assignment-time availability without claiming a recovery probe.

        Round-2 V3: a probe claim past its wall-clock TTL no longer blocks.
        """
        if self.endpoint_correlation_active(endpoint):
            return False
        state = self._cooldowns.get((base_model, provider))
        if state is None:
            return True
        if self._clock() < state.until:
            return False
        return not state.probe_in_flight or self._probe_claim_stale(state)

    def admission(
        self, base_model: str, provider: str, *, endpoint: str | None = None
    ) -> tuple[bool, bool, int | None]:
        """Return admission, probe status, and an owner token for expired probes.

        Round-2 V3: a probe claim past its wall-clock TTL no longer blocks —
        it is re-claimed under a NEW owner token. The orphaned old token can
        never release the new claim (release stays token-owned).
        """
        if self.endpoint_correlation_active(endpoint):
            return False, False, None
        state = self._cooldowns.get((base_model, provider))
        if state is None:
            return True, False, None
        now = self._clock()
        if now < state.until:
            return False, False, None
        if state.probe_in_flight and not self._probe_claim_stale(state):
            return False, False, None
        self._next_probe_token += 1
        state.probe_in_flight = True
        state.probe_token = self._next_probe_token
        state.claimed_at = now
        return True, True, state.probe_token

    def release_probe(self, base_model: str, provider: str, token: int) -> None:
        """Release an unattempted probe only when its owner token still matches."""
        state = self._cooldowns.get((base_model, provider))
        if state is not None and state.probe_token == token:
            state.probe_in_flight = False
            state.probe_token = None
            state.claimed_at = None

    def cooldown_until(self, base_model: str, provider: str) -> float | None:
        state = self._cooldowns.get((base_model, provider))
        return state.until if state is not None else None

    def probe_in_flight(self, base_model: str, provider: str) -> bool:
        """Read-only: whether a single-flight recovery probe is claimed.

        R2's eligibility snapshot reports probe occupancy so a read-only
        scheduling check can wait/recheck within the episode instead of
        consuming a retry — without ever claiming or releasing the probe.

        Round-2 V3: a claim past its wall-clock TTL is dead and reports as
        not in flight (read-only: the claim itself is never mutated).
        """
        state = self._cooldowns.get((base_model, provider))
        if state is None or not state.probe_in_flight:
            return False
        return not self._probe_claim_stale(state)


def compatibility_exclusion(
    *,
    config: object,
    model: ModelConfig,
    history: Sequence[LLMMessage],
    thinking: ThinkingLevel,
    thinking_explicit: bool,
    images_only: bool = False,
) -> tuple[ExclusionReason, str] | None:
    """Return why a concrete deployment cannot serve the exact request."""
    if any(message.images for message in history) and not model.supports_images:
        return (
            ExclusionReason.IMAGES_UNSUPPORTED,
            "Retained history contains images unsupported by the model",
        )
    provider = config.get_provider_for_model(model)  # type: ignore[attr-defined]
    style = provider.api_style
    for message in () if images_only else history:
        payloads = message.reasoning_payloads or ()
        if not payloads:
            continue
        if style == "anthropic":
            if any(
                block.get("type") not in REASONING_BLOCK_TYPES for block in payloads
            ):
                return (
                    ExclusionReason.REASONING_UNSUPPORTED,
                    "Retained history contains incompatible provider reasoning blocks",
                )
        elif style == "openai-responses":
            if (
                any(block.get("type") != "reasoning" for block in payloads)
                and not message.reasoning_content
            ):
                return (
                    ExclusionReason.REASONING_UNSUPPORTED,
                    "Retained history contains non-convertible provider reasoning blocks",
                )
        elif not message.reasoning_content:
            return (
                ExclusionReason.REASONING_UNSUPPORTED,
                "Retained history contains non-convertible provider reasoning blocks",
            )

    levels = get_thinking_levels(str(provider.backend), provider.api_style, model.name)
    declared = model.supported_thinking_levels
    incompatible = not images_only and (
        (levels is None and (declared is not None or thinking_explicit))
        or (
            levels is not None
            and declared is not None
            and any(level not in levels for level in declared)
        )
        or (levels is not None and thinking not in levels)
        or (declared is not None and thinking not in declared)
    )
    if incompatible:
        return (
            ExclusionReason.THINKING_UNSUPPORTED,
            "Configured thinking level is unsupported by the selected model",
        )
    return None


def cooldown_override_bypasses(
    override_permit: OverridePermit | None,
    override_ledger: CooldownOverrideLedger | None,
    endpoint_identity: str | None,
) -> bool:
    """Whether a typed forced-attempt permit bypasses one candidate's cooldowns.

    The bypass is endpoint-scoped and fail-closed: the permit must be the
    ledger's live reservation for its own episode — a wrong-endpoint,
    expired, released, or already-used permit never bypasses — and the
    candidate must serve the permit's canonical endpoint. Only the cooldown
    predicates (deployment cooldown, endpoint-wide correlation) are
    bypassed; every other admission check still applies, and the bypass
    never mutates the registry.
    """
    if override_permit is None or override_ledger is None:
        return False
    if endpoint_identity is None or endpoint_identity != override_permit.endpoint:
        return False
    return override_ledger.permit_outstanding(override_permit)


def eligible_deployments(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915
    *,
    snapshot: CatalogSnapshot,
    committed: CommittedModelIdentity,
    registry: AvailabilityRegistry,
    config: object,
    history: Sequence[LLMMessage] = (),
    thinking: ThinkingLevel,
    thinking_explicit: bool = True,
    compaction_base: str | None = None,
    compaction_only: bool = False,
    allowed_models: Sequence[str] = (),
    claim_probes: bool = True,
    override_permit: OverridePermit | None = None,
    override_ledger: CooldownOverrideLedger | None = None,
) -> EligibilityResult:
    """Resolve ordered candidates for exactly the committed canonical base.

    ``claim_probes`` (WP4): the default claims the registry's single-flight
    recovery probe per surviving candidate (the failover path's behavior).
    ``claim_probes=False`` is the READ-ONLY eligibility view the automatic
    recovery controller uses for scheduling: no probe is claimed or released,
    so the controller can never consume another caller's probe or strand a
    token-owned claim — availability is read through ``is_available`` only.

    ``override_permit``/``override_ledger`` (C2 typed permits): a permit
    minted by the root ``CooldownOverrideLedger`` bypasses ONLY the
    cooldown predicates — the endpoint-wide correlation check and the
    deployment admission/availability check — for candidates serving the
    permit's canonical endpoint. Every other admission check (disabled,
    allowlist, compaction, compatibility) still applies, and a
    wrong/expired/released/used permit never bypasses. The bypass never
    mutates the registry: a bypassed candidate is admitted with no probe
    claim and no cooldown is cleared.

    Round-2 V3, claim safety: every probe claimed during the pass is
    tracked, and an out-of-band error (exception or cancellation) between
    claiming a probe and handing its candidate to the caller releases them
    all (release stays token-owned) before re-raising — no claim can strand
    and probe-block a deployment until a user reset_base. Claims also
    carry a wall-clock TTL (see ``AvailabilityRegistry``), so even a claim
    stranded outside this process stops blocking and can be re-claimed.
    """
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema

    if not isinstance(config, ChartreuxConfigSchema):
        raise TypeError("config must be a ChartreuxConfigSchema")
    definition = snapshot.catalog.models.get(committed.base_model)
    if definition is None or definition.disabled:
        raise AllDeploymentsUnavailableError(committed.base_model, ())

    resolver = ModelResolver(snapshot)
    candidates: list[EligibleDeployment] = []
    exclusions: list[DeploymentExclusion] = []
    canonical_compaction_base = (
        resolver.canonicalize(compaction_base) if compaction_base else None
    )
    compaction_definition = (
        snapshot.catalog.models.get(canonical_compaction_base)
        if canonical_compaction_base
        else None
    )

    # Round-2 V3: every probe claimed during this pass, in claim order.
    claimed_probes: list[tuple[str, int]] = []
    try:  # noqa: PLR1702
        for deployment in definition.deployments:
            provider = snapshot.catalog.providers.get(deployment.provider)
            exclusion: DeploymentExclusion | None = None
            endpoint_identity = (
                canonical_endpoint_identity(provider.api_base)
                if provider is not None
                else None
            )
            cooldown_bypassed = cooldown_override_bypasses(
                override_permit, override_ledger, endpoint_identity
            )
            if (
                endpoint_identity is not None
                and not cooldown_bypassed
                and registry.endpoint_correlation_active(endpoint_identity)
            ):
                # Endpoint-wide correlation cooldown (WP1): cross-model rate-limit
                # evidence on ONE endpoint excludes every deployment of that
                # endpoint, checked before the per-deployment chain.
                exclusion = DeploymentExclusion(
                    committed.base_model,
                    deployment.provider,
                    deployment.name,
                    ExclusionReason.ENDPOINT_COOLDOWN,
                    "endpoint-wide correlation cooldown (cross-model rate-limit evidence)",
                )
            elif deployment.disabled or provider is None or provider.disabled:
                exclusion = DeploymentExclusion(
                    committed.base_model,
                    deployment.provider,
                    deployment.name,
                    ExclusionReason.DISABLED,
                )
            elif not resolver._allowed(
                committed.base_model, definition, deployment, allowed_models
            ):
                exclusion = DeploymentExclusion(
                    committed.base_model,
                    deployment.provider,
                    deployment.name,
                    ExclusionReason.ALLOWLIST_EXCLUDED,
                )
            elif compaction_base and (
                compaction_definition is None
                or compaction_definition.disabled
                or not any(
                    item.provider == deployment.provider
                    and not item.disabled
                    and not snapshot.catalog.providers[item.provider].disabled
                    for item in compaction_definition.deployments
                )
            ):
                exclusion = DeploymentExclusion(
                    committed.base_model,
                    deployment.provider,
                    deployment.name,
                    ExclusionReason.COMPACTION_INCOMPATIBLE,
                )
            else:
                if cooldown_bypassed:
                    # C2 typed permit: ONLY this candidate's cooldown
                    # predicates are bypassed. The candidate is admitted
                    # with no probe claim and the registry is untouched —
                    # no cooldown is cleared, no probe is claimed or
                    # released (the forced attempt is metered by the
                    # override ledger, not the single-flight probe).
                    admitted, probe, probe_token = True, False, None
                elif claim_probes:
                    admitted, probe, probe_token = registry.admission(
                        committed.base_model,
                        deployment.provider,
                        endpoint=endpoint_identity,
                    )
                    if probe:
                        assert probe_token is not None
                        claimed_probes.append((deployment.provider, probe_token))
                else:
                    # WP4 read-only eligibility: never claim (or release) the
                    # single-flight recovery probe — scheduling checks stay
                    # token-owned by the callers that actually attempt.
                    admitted = registry.is_available(
                        committed.base_model,
                        deployment.provider,
                        endpoint=endpoint_identity,
                    )
                    probe, probe_token = False, None
                if not admitted:
                    exclusion = DeploymentExclusion(
                        committed.base_model,
                        deployment.provider,
                        deployment.name,
                        ExclusionReason.COOLDOWN,
                    )
                else:
                    resolved = ResolvedModel(
                        committed.base_model,
                        definition,
                        deployment,
                        provider,
                        snapshot.revision,
                    )
                    model = resolved.materialize(
                        auto_compact_threshold=config.auto_compact_threshold,
                        thinking=thinking,
                        validate_thinking=False,
                    )
                    problem = compatibility_exclusion(
                        config=config,
                        model=model,
                        history=history,
                        thinking=thinking,
                        thinking_explicit=thinking_explicit,
                        images_only=compaction_only,
                    )
                    if problem is not None:
                        reason, detail = problem
                        exclusion = DeploymentExclusion(
                            committed.base_model,
                            deployment.provider,
                            deployment.name,
                            reason,
                            detail,
                        )
                    elif compaction_definition is not None:
                        compaction_deployment = next(
                            item
                            for item in compaction_definition.deployments
                            if item.provider == deployment.provider
                            and not item.disabled
                        )
                        compaction_resolved = ResolvedModel(
                            canonical_compaction_base or "",
                            compaction_definition,
                            compaction_deployment,
                            snapshot.catalog.providers[compaction_deployment.provider],
                            snapshot.revision,
                        )
                        compaction_thinking = (
                            thinking
                            if compaction_only
                            else config.thinking_overrides.get(
                                canonical_compaction_base or "",
                                compaction_definition.thinking,
                            )
                        )
                        compaction_model = compaction_resolved.materialize(
                            auto_compact_threshold=config.auto_compact_threshold,
                            thinking=compaction_thinking,
                            validate_thinking=False,
                        )
                        compaction_problem = compatibility_exclusion(
                            config=config,
                            model=compaction_model,
                            history=history,
                            thinking=compaction_model.thinking or "off",
                            thinking_explicit=False,
                        )
                        if compaction_problem is not None:
                            _, detail = compaction_problem
                            exclusion = DeploymentExclusion(
                                committed.base_model,
                                deployment.provider,
                                deployment.name,
                                ExclusionReason.COMPACTION_INCOMPATIBLE,
                                detail,
                            )
                    if exclusion is None:
                        candidates.append(
                            EligibleDeployment(resolved, probe, probe_token)
                        )
                    elif probe:
                        assert probe_token is not None
                        registry.release_probe(
                            committed.base_model, deployment.provider, probe_token
                        )
            if exclusion is not None:
                exclusions.append(exclusion)
    except BaseException:
        # Round-2 V3: an out-of-band error (exception or cancellation)
        # between claiming a probe and handing its candidate to the caller
        # releases every claim made during this call — release stays
        # token-owned, so a foreign token never releases another claim —
        # before re-raising: no claim can strand.
        for claimed_provider, claimed_token in claimed_probes:
            registry.release_probe(
                committed.base_model, claimed_provider, claimed_token
            )
        raise

    if not candidates:
        raise AllDeploymentsUnavailableError(committed.base_model, exclusions)
    return EligibilityResult(tuple(candidates), tuple(exclusions))


@dataclass(frozen=True, slots=True)
class AvailabilitySnapshot:
    """The NON-MUTATING eligibility snapshot (R2 root recovery).

    ``available``: at least one deployment of the committed base is eligible
    right now. ``permanent``: every deployment is excluded for a PERMANENT
    reason (disabled/allowlist/images/thinking/reasoning/compaction) —
    recovery must not be scheduled at all. ``endpoint_blocked``: every
    deployment is excluded by endpoint-wide correlation. ``probe_in_flight``:
    some deployment's single-flight recovery probe is claimed by another
    caller — wait/recheck, never claim it for a read-only check.
    ``next_check_delay``: seconds until the earliest temporary exclusion
    (deployment cooldown, endpoint cooldown) may lift; 0 when the only
    blocker is probe occupancy (the caller applies its recheck floor).
    """

    available: bool
    permanent: bool
    endpoint_blocked: bool
    probe_in_flight: bool
    next_check_delay: float
    exclusions: tuple[DeploymentExclusion, ...] = ()


#: Exclusion reasons that are TEMPORARY (time-based) — recovery waits for
#: them instead of stopping. Everything else is permanent incompatibility.
TEMPORARY_EXCLUSION_REASONS = frozenset({
    ExclusionReason.COOLDOWN,
    ExclusionReason.ENDPOINT_COOLDOWN,
})


def availability_snapshot(
    *,
    snapshot: CatalogSnapshot,
    committed: CommittedModelIdentity,
    registry: AvailabilityRegistry,
    config: object,
    history: Sequence[LLMMessage] = (),
    thinking: ThinkingLevel,
    compaction_base: str | None = None,
    allowed_models: Sequence[str] = (),
) -> AvailabilitySnapshot:
    """Read the same-base eligibility WITHOUT mutating any registry state.

    R2 (plan state.md): the root recovery gate's cooldown-aware scheduling
    uses this snapshot instead of ``eligible_deployments``' claim path. It
    never claims or releases a single-flight probe (``claim_probes=False``),
    never records a failure or success, and never clears a cooldown — a
    read-only check can never consume another caller's probe or strand a
    token-owned claim. Endpoint-wide correlation cooldowns are honored
    through the same exclusions the admission path sees (checked FIRST).
    """
    try:
        result = eligible_deployments(
            snapshot=snapshot,
            committed=committed,
            registry=registry,
            config=config,
            history=history,
            thinking=thinking,
            compaction_base=compaction_base,
            allowed_models=allowed_models,
            claim_probes=False,
        )
    except AllDeploymentsUnavailableError as exc:
        return _snapshot_from_exclusions(committed, registry, snapshot, exc.exclusions)
    return AvailabilitySnapshot(
        available=True,
        permanent=False,
        endpoint_blocked=False,
        probe_in_flight=False,
        next_check_delay=0.0,
        exclusions=result.exclusions,
    )


def _snapshot_from_exclusions(
    committed: CommittedModelIdentity,
    registry: AvailabilityRegistry,
    snapshot: CatalogSnapshot,
    exclusions: Sequence[DeploymentExclusion],
) -> AvailabilitySnapshot:
    """Derive the wait/permanent/endpoint-blocked snapshot from exclusions."""
    now = registry._clock()
    gaps: list[float] = []
    any_temporary = False
    all_endpoint = bool(exclusions)
    probe_in_flight = False
    for exclusion in exclusions:
        if exclusion.reason not in TEMPORARY_EXCLUSION_REASONS:
            all_endpoint = False
            continue
        any_temporary = True
        if exclusion.reason is ExclusionReason.ENDPOINT_COOLDOWN:
            provider = snapshot.catalog.providers.get(exclusion.provider)
            endpoint = canonical_endpoint_identity(getattr(provider, "api_base", None))
            until = registry.endpoint_cooldown_until(endpoint)
        else:
            all_endpoint = False
            until = registry.cooldown_until(committed.base_model, exclusion.provider)
            if registry.probe_in_flight(committed.base_model, exclusion.provider):
                probe_in_flight = True
        gaps.append(max(0.0, until - now) if until is not None else 0.0)
    if exclusions and not any_temporary:
        # Every deployment is excluded for a permanent reason: recovery must
        # not be scheduled (all-deployments-unavailable rejects permanent
        # incompatibility).
        return AvailabilitySnapshot(
            available=False,
            permanent=True,
            endpoint_blocked=False,
            probe_in_flight=False,
            next_check_delay=0.0,
            exclusions=tuple(exclusions),
        )
    return AvailabilitySnapshot(
        available=False,
        permanent=False,
        endpoint_blocked=all_endpoint,
        probe_in_flight=probe_in_flight,
        next_check_delay=min(gaps) if gaps else 0.0,
        exclusions=tuple(exclusions),
    )
