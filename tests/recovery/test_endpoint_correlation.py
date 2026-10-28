"""WP1 — endpoint-keyed failure correlation in the shared availability registry.

Fast-429 exhaustion and cross-model endpoint correlation are represented
WITHOUT log parsing: the registry records bounded endpoint events keyed on
the canonical endpoint identity (userinfo stripped) and arms an
endpoint-wide cooldown only on cross-model rate-limit evidence — repeated
failures with an identical structured signature across enough distinct base
models on ONE endpoint. A single isolated timeout never broadens to
endpoint-wide, and endpoint cooldowns are time-based: no single success
clears them (the fleet-wide-clear defect class).

All clocks are the opt-in ``fake_clock`` fixture; no real sleeps.
"""

from __future__ import annotations

import pytest

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.llm.failures import FailureCategory, FailureInfo
from chartreux.core.model_catalog.availability import (
    AllDeploymentsUnavailableError,
    AvailabilityRegistry,
    ExclusionReason,
    eligible_deployments,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.recovery_models import (
    CallEvidence,
    FailureClass,
    RecoveryEligibility,
    attach_call_evidence,
    build_failure_envelope,
    canonical_endpoint_identity,
)
from chartreux.core.recovery_observation import observe_exception
from chartreux.core.session_types import CommittedModelIdentity
from tests.recovery.conftest import FakeClock

ENDPOINT_URL = "https://api.shared.test/v1"
ENDPOINT_IDENTITY = "api.shared.test"
RATE_LIMIT_SIGNATURE = "rate_limit:429:rate_limit:rate_limit"


def _rate_limit() -> FailureInfo:
    return FailureInfo(FailureCategory.RATE_LIMIT, status=429)


def _record(
    registry: AvailabilityRegistry,
    base_model: str,
    provider: str,
    *,
    endpoint: str = ENDPOINT_URL,
    signature: str = RATE_LIMIT_SIGNATURE,
    category: FailureCategory = FailureCategory.RATE_LIMIT,
) -> None:
    registry.record_failure(
        base_model,
        provider,
        FailureInfo(
            category, status=429 if category is FailureCategory.RATE_LIMIT else None
        ),
        endpoint=endpoint,
        signature=signature,
    )


def test_cross_model_rate_limit_signature_arms_endpoint_cooldown(
    fake_clock: FakeClock,
) -> None:
    registry = AvailabilityRegistry(
        clock=fake_clock, initial_cooldown=30.0, endpoint_cooldown=45.0
    )
    _record(registry, "model-a", "provider-1")
    assert not registry.endpoint_correlation_active(ENDPOINT_IDENTITY)
    _record(registry, "model-b", "provider-2")
    assert registry.endpoint_correlation_active(ENDPOINT_IDENTITY)
    assert registry.endpoint_cooldown_until(ENDPOINT_IDENTITY) == 45.0
    # The endpoint cooldown gates assignment-time availability and admission.
    assert (
        registry.is_available("model-c", "provider-3", endpoint=ENDPOINT_IDENTITY)
        is False
    )
    assert registry.admission("model-c", "provider-3", endpoint=ENDPOINT_IDENTITY) == (
        False,
        False,
        None,
    )
    # A different endpoint is unaffected.
    assert registry.is_available("model-c", "provider-3", endpoint="api.other.test")


def test_isolated_timeout_never_broadens_to_endpoint_wide(
    fake_clock: FakeClock,
) -> None:
    registry = AvailabilityRegistry(clock=fake_clock)
    _record(
        registry,
        "model-a",
        "provider-1",
        signature="timeout:None::",
        category=FailureCategory.TIMEOUT,
    )
    _record(
        registry,
        "model-b",
        "provider-2",
        signature="timeout:None::",
        category=FailureCategory.TIMEOUT,
    )
    assert not registry.endpoint_correlation_active(ENDPOINT_IDENTITY)
    # A never-failed deployment on the same endpoint stays available: the
    # isolated timeouts armed only their own per-deployment cooldowns.
    assert registry.is_available("model-c", "provider-3", endpoint=ENDPOINT_IDENTITY)


def test_single_model_repeated_429_does_not_correlate(fake_clock: FakeClock) -> None:
    """Repeated same-model 429s are a per-deployment cooldown, not an
    endpoint signal: correlation requires >= min_models distinct bases.
    """
    registry = AvailabilityRegistry(clock=fake_clock)
    for _ in range(3):
        _record(registry, "model-a", "provider-1")
    assert not registry.endpoint_correlation_active(ENDPOINT_IDENTITY)


def test_different_signatures_do_not_correlate(fake_clock: FakeClock) -> None:
    registry = AvailabilityRegistry(clock=fake_clock)
    _record(registry, "model-a", "provider-1", signature="rate_limit:429:a:")
    _record(registry, "model-b", "provider-2", signature="rate_limit:429:b:")
    assert not registry.endpoint_correlation_active(ENDPOINT_IDENTITY)


def test_empty_signature_never_correlates(fake_clock: FakeClock) -> None:
    registry = AvailabilityRegistry(clock=fake_clock)
    _record(registry, "model-a", "provider-1", signature="")
    _record(registry, "model-b", "provider-2", signature="")
    assert not registry.endpoint_correlation_active(ENDPOINT_IDENTITY)


def test_window_expiry_decays(fake_clock: FakeClock) -> None:
    """Events outside the correlation window no longer count, and the
    armed cooldown itself expires on its own clock.
    """
    registry = AvailabilityRegistry(
        clock=fake_clock, endpoint_correlation_window=10.0, endpoint_cooldown=5.0
    )
    _record(registry, "model-a", "provider-1")
    fake_clock.advance(20.0)
    # The stale event is pruned: the second failure is alone in the window.
    _record(registry, "model-b", "provider-2")
    assert not registry.endpoint_correlation_active(ENDPOINT_IDENTITY)

    registry2 = AvailabilityRegistry(
        clock=fake_clock, endpoint_correlation_window=300.0, endpoint_cooldown=5.0
    )
    _record(registry2, "model-a", "provider-1")
    _record(registry2, "model-b", "provider-2")
    assert registry2.endpoint_correlation_active(ENDPOINT_IDENTITY)
    fake_clock.advance(6.0)
    assert not registry2.endpoint_correlation_active(ENDPOINT_IDENTITY)


def test_record_success_does_not_clear_endpoint_cooldown(fake_clock: FakeClock) -> None:
    """Endpoint correlation cooldowns are time-based endpoint evidence; a
    single success must not broaden-clear them (fleet-wide-clear defect).
    """
    registry = AvailabilityRegistry(clock=fake_clock, endpoint_cooldown=45.0)
    _record(registry, "model-a", "provider-1")
    _record(registry, "model-b", "provider-2")
    assert registry.endpoint_correlation_active(ENDPOINT_IDENTITY)
    registry.record_success("model-a", "provider-1")
    assert registry.endpoint_correlation_active(ENDPOINT_IDENTITY)
    # The per-deployment cooldown IS cleared by the success.
    assert registry.cooldown_until("model-a", "provider-1") is None


def test_reset_base_does_not_clear_endpoint_cooldown(fake_clock: FakeClock) -> None:
    registry = AvailabilityRegistry(clock=fake_clock, endpoint_cooldown=45.0)
    _record(registry, "model-a", "provider-1")
    _record(registry, "model-b", "provider-2")
    assert registry.endpoint_correlation_active(ENDPOINT_IDENTITY)
    registry.reset_base("model-a")
    assert registry.endpoint_correlation_active(ENDPOINT_IDENTITY)


def test_userinfo_and_case_normalized_keys(fake_clock: FakeClock) -> None:
    """The endpoint key space cannot split: raw URLs with userinfo/case
    canonicalize to the same identity the lookups use.
    """
    registry = AvailabilityRegistry(clock=fake_clock)
    registry.record_failure(
        "model-a",
        "provider-1",
        _rate_limit(),
        endpoint="https://user:secret@API.Shared.Test/v1",
        signature=RATE_LIMIT_SIGNATURE,
    )
    registry.record_failure(
        "model-b",
        "provider-2",
        _rate_limit(),
        endpoint="https://api.shared.test/v1",
        signature=RATE_LIMIT_SIGNATURE,
    )
    assert registry.endpoint_correlation_active(ENDPOINT_IDENTITY)


def test_registry_validation_rejects_bad_correlation_params() -> None:
    with pytest.raises(ValueError, match="endpoint_correlation_window"):
        AvailabilityRegistry(endpoint_correlation_window=0)
    with pytest.raises(ValueError, match="endpoint_correlation_min_events"):
        AvailabilityRegistry(endpoint_correlation_min_events=0)
    with pytest.raises(ValueError, match="endpoint_correlation_min_models"):
        AvailabilityRegistry(endpoint_correlation_min_models=0)
    with pytest.raises(ValueError, match="endpoint_cooldown"):
        AvailabilityRegistry(endpoint_cooldown=-1)


def test_canonical_endpoint_identity_normalizes_default_ports() -> None:
    """Review R9: the literal port never splits endpoint identity — an
    explicit DEFAULT port (443 for https, 80 for http) is dropped, so
    ``https://example.test`` and ``https://example.test:443`` are ONE
    identity (same-endpoint substitution cannot pass independence on a port
    spelling; correlation state cannot split on it).
    """
    assert canonical_endpoint_identity("https://example.test") == "example.test"
    assert canonical_endpoint_identity("https://example.test:443") == "example.test"
    assert canonical_endpoint_identity("http://example.test") == "example.test"
    assert canonical_endpoint_identity("http://example.test:80") == "example.test"
    # A NON-default port is part of the identity.
    assert (
        canonical_endpoint_identity("https://example.test:8443") == "example.test:8443"
    )
    assert (
        canonical_endpoint_identity("http://example.test:8080") == "example.test:8080"
    )
    # IPv6 hosts stay bracketed; userinfo/case still normalize.
    assert canonical_endpoint_identity("https://[2001:DB8::1]") == "[2001:db8::1]"
    assert canonical_endpoint_identity("https://[2001:db8::1]:443") == "[2001:db8::1]"
    assert (
        canonical_endpoint_identity("https://[2001:db8::1]:8443")
        == "[2001:db8::1]:8443"
    )
    assert (
        canonical_endpoint_identity("https://user:pw@EXAMPLE.test:443/v1")
        == "example.test"
    )
    # Unavailable/unparsable stays null — never a guess.
    assert canonical_endpoint_identity(None) is None
    assert canonical_endpoint_identity("") is None
    assert canonical_endpoint_identity("not a url") is None
    assert canonical_endpoint_identity("https://example.test:abc") is None


def test_port_spelling_never_splits_endpoint_correlation(fake_clock: FakeClock) -> None:
    """Review R9: failures recorded through differently-spelled URLs of the
    SAME endpoint (default-port spelling) correlate on one identity.
    """
    registry = AvailabilityRegistry(clock=fake_clock)
    registry.record_failure(
        "model-a",
        "provider-1",
        _rate_limit(),
        endpoint="https://api.shared.test:443/v1",
        signature=RATE_LIMIT_SIGNATURE,
    )
    registry.record_failure(
        "model-b",
        "provider-2",
        _rate_limit(),
        endpoint="https://api.shared.test/v1",
        signature=RATE_LIMIT_SIGNATURE,
    )
    assert registry.endpoint_correlation_active(ENDPOINT_IDENTITY)


def test_late_normal_success_never_settles_an_in_flight_probe(
    fake_clock: FakeClock,
) -> None:
    """Review R13: probe settlement is token-owned — a late normal
    attempt's success lifts the cooldown but never deletes a NEWER
    in-flight recovery probe's claim (single-flight holds).
    """
    registry = AvailabilityRegistry(clock=fake_clock, initial_cooldown=30.0)
    registry.record_failure("model-a", "provider-1", _rate_limit())
    fake_clock.advance(31.0)
    # The cooldown expired: admission claims the single-flight probe.
    admitted, probe, token = registry.admission("model-a", "provider-1")
    assert (admitted, probe) == (True, True)
    assert token is not None
    # A late NORMAL attempt succeeds (no probe token): the cooldown lifts,
    # but the probe claim survives — single-flight holds.
    registry.record_success("model-a", "provider-1")
    assert registry.probe_in_flight("model-a", "provider-1")
    assert registry.admission("model-a", "provider-1") == (False, False, None)
    until = registry.cooldown_until("model-a", "provider-1")
    assert until is not None and until <= fake_clock()
    # Only the owner's own settlement releases the probe.
    registry.record_success("model-a", "provider-1", probe_token=token)
    assert not registry.probe_in_flight("model-a", "provider-1")
    assert registry.cooldown_until("model-a", "provider-1") is None
    assert registry.is_available("model-a", "provider-1")


def test_late_normal_failure_never_settles_an_in_flight_probe(
    fake_clock: FakeClock,
) -> None:
    """Review R13: probe settlement is token-owned — a late normal
    attempt's failure extends the cooldown but never replaces a NEWER
    in-flight recovery probe's claim.
    """
    registry = AvailabilityRegistry(clock=fake_clock, initial_cooldown=30.0)
    registry.record_failure("model-a", "provider-1", _rate_limit())
    fake_clock.advance(31.0)
    _admitted, _probe, token = registry.admission("model-a", "provider-1")
    assert token is not None
    # A late NORMAL attempt fails (no probe token): the cooldown extends,
    # but the probe claim survives — single-flight holds.
    registry.record_failure("model-a", "provider-1", _rate_limit())
    assert registry.probe_in_flight("model-a", "provider-1")
    assert registry.admission("model-a", "provider-1") == (False, False, None)
    until = registry.cooldown_until("model-a", "provider-1")
    assert until is not None and until > 31.0
    # The owner's own failure settles (and releases) the probe.
    registry.record_failure("model-a", "provider-1", _rate_limit(), probe_token=token)
    assert not registry.probe_in_flight("model-a", "provider-1")


# ---------------------------------------------------------------------------
# eligible_deployments: the endpoint cooldown excludes before the chain
# ---------------------------------------------------------------------------


def _catalog() -> CatalogSnapshot:
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test-first": {"api_base": ENDPOINT_URL},
                "test-second": {"api_base": "https://second.test"},
            },
            "models": {
                "base": {
                    "deployments": [
                        {
                            "provider": "test-first",
                            "name": "first",
                            "supported_thinking_levels": ["off"],
                        },
                        {
                            "provider": "test-second",
                            "name": "second",
                            "supported_thinking_levels": ["off"],
                        },
                    ]
                }
            },
            "roles": {},
        }),
        "test",
    )


def _config(snapshot: CatalogSnapshot) -> ChartreuxConfigSchema:
    return ChartreuxConfigSchema(active_model="base").attach_catalog_snapshot(snapshot)


def _committed() -> CommittedModelIdentity:
    return CommittedModelIdentity(
        base_model="base",
        provider="test-first",
        wire_name="first",
        catalog_revision="test",
    )


def test_eligible_deployments_excludes_endpoint_cooldown(fake_clock: FakeClock) -> None:
    """An armed endpoint cooldown excludes the endpoint's deployments BEFORE
    the per-deployment chain, with the endpoint-cooldown exclusion reason.
    """
    snapshot = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    _record(registry, "model-a", "provider-1")
    _record(registry, "model-b", "provider-2")
    assert registry.endpoint_correlation_active(ENDPOINT_IDENTITY)

    result = eligible_deployments(
        snapshot=snapshot,
        committed=_committed(),
        registry=registry,
        config=_config(snapshot),
        thinking="off",
    )
    # Only the second endpoint's deployment survives.
    assert len(result.candidates) == 1
    assert result.candidates[0].resolved.identity.provider == "test-second"
    assert len(result.exclusions) == 1
    exclusion = result.exclusions[0]
    assert exclusion.reason is ExclusionReason.ENDPOINT_COOLDOWN
    assert exclusion.provider == "test-first"


def test_all_deployments_unavailable_carries_endpoint_cooldown(
    fake_clock: FakeClock,
) -> None:
    """When every deployment shares the correlated endpoint, the exhaustion
    error's exclusions carry endpoint-cooldown — the envelope's
    exclusion_reasons source.
    """
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {"test-first": {"api_base": ENDPOINT_URL}},
            "models": {
                "base": {
                    "deployments": [
                        {
                            "provider": "test-first",
                            "name": "first",
                            "supported_thinking_levels": ["off"],
                        }
                    ]
                }
            },
            "roles": {},
        }),
        "test",
    )
    registry = AvailabilityRegistry(clock=fake_clock)
    _record(registry, "model-a", "provider-1")
    _record(registry, "model-b", "provider-2")

    with pytest.raises(AllDeploymentsUnavailableError) as exc_info:
        eligible_deployments(
            snapshot=snapshot,
            committed=_committed(),
            registry=registry,
            config=_config(snapshot),
            thinking="off",
        )
    assert len(exc_info.value.exclusions) == 1
    assert exc_info.value.exclusions[0].reason is ExclusionReason.ENDPOINT_COOLDOWN
    assert exc_info.value.exclusions[0].reason.value == "endpoint-cooldown"


def test_endpoint_cooldown_exclusions_classify_transient_launch_failure() -> None:
    """C7: a pure endpoint-cooldown exclusion set is a transient launch
    exclusion — the same time-based block as a per-deployment cooldown —
    while a mixed set containing a permanent reason stays prohibited.
    """
    admission_error = AllDeploymentsUnavailableError("base", ())
    attach_call_evidence(admission_error, CallEvidence.admission_phase())
    observation = observe_exception(admission_error, phase="turn", turn_id="t")

    endpoint_only = build_failure_envelope(
        observation=observation,
        agent_id="a",
        run_id="r",
        exclusion_reasons=["endpoint-cooldown"],
    )
    assert endpoint_only.failure.failure_class is FailureClass.LAUNCH_FAILURE
    assert endpoint_only.recovery.eligibility is RecoveryEligibility.ELIGIBLE
    assert "transient_launch_failure" in endpoint_only.recovery.reason_codes
    assert "endpoint_cooldown_exclusion" in endpoint_only.failure.evidence_codes

    mixed = build_failure_envelope(
        observation=observation,
        agent_id="a",
        run_id="r",
        exclusion_reasons=["endpoint-cooldown", "disabled"],
    )
    assert mixed.recovery.eligibility is RecoveryEligibility.PROHIBITED
    assert "permanent_launch_exclusion" in mixed.recovery.reason_codes
    assert "permanent_exclusion" in mixed.failure.evidence_codes


def test_endpoint_cooldown_checked_before_disabled(fake_clock: FakeClock) -> None:
    """The endpoint correlation check precedes the per-deployment chain."""
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {"test-first": {"api_base": ENDPOINT_URL}},
            "models": {
                "base": {
                    "deployments": [
                        {
                            "provider": "test-first",
                            "name": "first",
                            "supported_thinking_levels": ["off"],
                            "disabled": True,
                        }
                    ]
                }
            },
            "roles": {},
        }),
        "test",
    )
    registry = AvailabilityRegistry(clock=fake_clock)
    _record(registry, "model-a", "provider-1")
    _record(registry, "model-b", "provider-2")

    with pytest.raises(AllDeploymentsUnavailableError) as exc_info:
        eligible_deployments(
            snapshot=snapshot,
            committed=_committed(),
            registry=registry,
            config=_config(snapshot),
            thinking="off",
        )
    assert exc_info.value.exclusions[0].reason is ExclusionReason.ENDPOINT_COOLDOWN


def test_read_only_eligibility_never_claims_or_releases_probes(
    fake_clock: FakeClock,
) -> None:
    """WP4: ``claim_probes=False`` is the controller's non-mutating view.

    The read-only eligibility evaluation excludes endpoint-cooldown and
    per-deployment-cooldown deployments exactly like the claiming path, but
    it never calls ``admission`` (no single-flight probe is claimed) and
    never calls ``release_probe`` (no token-owned claim is touched). The
    returned candidates carry ``recovery_probe=False``.
    """
    snapshot = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    _record(registry, "model-a", "provider-1")
    _record(registry, "model-b", "provider-2")
    # A per-deployment cooldown on the second provider's deployment too.
    registry.record_failure("base", "test-second", FailureInfo(FailureCategory.TIMEOUT))

    claimed: list[tuple[str, str]] = []
    released: list[tuple[str, str, int]] = []
    original_admission = registry.admission
    original_release = registry.release_probe

    def spy_admission(
        base_model: str, provider: str, *, endpoint: str | None = None
    ) -> tuple[bool, bool, int | None]:
        claimed.append((base_model, provider))
        return original_admission(base_model, provider, endpoint=endpoint)

    def spy_release(base_model: str, provider: str, token: int) -> None:
        released.append((base_model, provider, token))
        original_release(base_model, provider, token)

    registry.admission = spy_admission  # type: ignore[method-assign]
    registry.release_probe = spy_release  # type: ignore[method-assign]

    # Read-only: the endpoint-cooldown deployment is excluded, the
    # cooldown-blocked deployment is excluded, and the surviving candidate
    # (none here — both are blocked) would carry no probe.
    with pytest.raises(AllDeploymentsUnavailableError) as exc_info:
        eligible_deployments(
            snapshot=snapshot,
            committed=_committed(),
            registry=registry,
            config=_config(snapshot),
            thinking="off",
            claim_probes=False,
        )
    reasons = [exclusion.reason for exclusion in exc_info.value.exclusions]
    assert reasons == [ExclusionReason.ENDPOINT_COOLDOWN, ExclusionReason.COOLDOWN]
    assert claimed == []
    assert released == []

    # After the cooldowns expire, the read-only view returns the candidates
    # without claiming a probe for them.
    fake_clock.advance(1000.0)
    result = eligible_deployments(
        snapshot=snapshot,
        committed=_committed(),
        registry=registry,
        config=_config(snapshot),
        thinking="off",
        claim_probes=False,
    )
    assert [
        candidate.resolved.identity.provider for candidate in result.candidates
    ] == ["test-first", "test-second"]
    assert all(
        candidate.recovery_probe is False and candidate.recovery_probe_token is None
        for candidate in result.candidates
    )
    assert claimed == []
    assert released == []

    # The claiming path (default) is unchanged: it calls ``admission`` for
    # every surviving deployment and claims the single-flight probe for the
    # one whose cooldown just expired (the probe mechanism only engages on
    # expired cooldowns; test-first has no cooldown state).
    claimed_result = eligible_deployments(
        snapshot=snapshot,
        committed=_committed(),
        registry=registry,
        config=_config(snapshot),
        thinking="off",
    )
    assert claimed_result.candidates[0].recovery_probe is False
    assert claimed_result.candidates[1].recovery_probe is True
    assert claimed == [("base", "test-first"), ("base", "test-second")]
    assert released == []
