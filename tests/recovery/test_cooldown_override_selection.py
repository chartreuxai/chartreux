"""C2 — typed permits at candidate selection.

A permit minted by the root ``CooldownOverrideLedger`` (C1) bypasses ONLY
the cooldown predicates at candidate selection — the deployment cooldown
and the endpoint-wide correlation cooldown — for candidates serving the
permit's canonical endpoint. Every other admission check still applies
(disabled, allowlist, compaction, thinking, history), a
wrong/expired/released/used permit never bypasses, and the bypass never
mutates the availability registry: a bypassed candidate is admitted with
no probe claim, no cooldown is cleared, and the permit stays outstanding
for the transport-start consumption (C3). The launch-time candidate
filter (``resolve_launch``) honors the same permit against its
availability predicate.

All clocks are the opt-in ``fake_clock`` fixture; no real sleeps.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from chartreux.core.agents.launch import resolve_launch
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import ModelConfig, ProviderConfig, ThinkingLevel
from chartreux.core.llm.failures import FailureCategory, FailureInfo
from chartreux.core.llm_models import (
    Backend,
    ImageAttachment,
    InlineImageSource,
    LLMMessage,
    Role,
)
from chartreux.core.model_catalog.availability import (
    AllDeploymentsUnavailableError,
    AvailabilityRegistry,
    ExclusionReason,
    eligible_deployments,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.recovery_override import (
    CooldownOverrideLedger,
    OverridePermit,
    OverridePhase,
)
from chartreux.core.session_types import CommittedModelIdentity
from chartreux.core.subagents import InvalidLaunchModelError, LaunchConfig
from tests.conftest import build_test_vibe_config
from tests.recovery.conftest import FakeClock
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

ENDPOINT_URL = "https://api.shared.test/v1"
ENDPOINT_IDENTITY = "api.shared.test"
OTHER_ENDPOINT_URL = "https://second.test"
OTHER_ENDPOINT_IDENTITY = "second.test"
RATE_LIMIT_SIGNATURE = "rate_limit:429:rate_limit:rate_limit"
LAUNCH_ENDPOINT_URL = "https://example.test"


def _catalog() -> CatalogSnapshot:
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test-first": {"api_base": ENDPOINT_URL},
                "test-second": {"api_base": OTHER_ENDPOINT_URL},
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


def _arm_deployment_cooldown(
    registry: AvailabilityRegistry, provider: str = "test-first"
) -> None:
    registry.record_failure("base", provider, FailureInfo(FailureCategory.TIMEOUT))


def _arm_endpoint_cooldown(registry: AvailabilityRegistry) -> None:
    for base_model, provider in (("model-a", "provider-1"), ("model-b", "provider-2")):
        registry.record_failure(
            base_model,
            provider,
            FailureInfo(FailureCategory.RATE_LIMIT, status=429),
            endpoint=ENDPOINT_URL,
            signature=RATE_LIMIT_SIGNATURE,
        )


def _reserved_permit(
    ledger: CooldownOverrideLedger, endpoint: str = ENDPOINT_URL
) -> OverridePermit:
    assert ledger.note_endpoint_failure(endpoint) is True
    outcome = ledger.reserve(endpoint)
    assert outcome.permit is not None
    return outcome.permit


def _select(
    snapshot: CatalogSnapshot, registry: AvailabilityRegistry, **kwargs: Any
) -> Any:
    return eligible_deployments(
        snapshot=snapshot,
        committed=_committed(),
        registry=registry,
        config=_config(snapshot),
        thinking="off",
        **kwargs,
    )


def _providers(result: Any) -> list[str]:
    return [candidate.resolved.identity.provider for candidate in result.candidates]


def test_deployment_cooldown_is_bypassed_for_the_matching_permit(
    fake_clock: FakeClock,
) -> None:
    snapshot = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_deployment_cooldown(registry)

    blocked = _select(snapshot, registry)
    assert _providers(blocked) == ["test-second"]
    assert [exclusion.reason for exclusion in blocked.exclusions] == [
        ExclusionReason.COOLDOWN
    ]

    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger)
    result = _select(snapshot, registry, override_permit=permit, override_ledger=ledger)
    assert _providers(result) == ["test-first", "test-second"]
    assert result.exclusions == ()
    # The bypassed candidate is admitted with no probe claim.
    assert result.candidates[0].recovery_probe is False
    assert result.candidates[0].recovery_probe_token is None


def test_endpoint_cooldown_is_bypassed_for_the_matching_permit(
    fake_clock: FakeClock,
) -> None:
    snapshot = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_endpoint_cooldown(registry)
    assert registry.endpoint_correlation_active(ENDPOINT_IDENTITY)

    blocked = _select(snapshot, registry)
    assert _providers(blocked) == ["test-second"]
    assert [exclusion.reason for exclusion in blocked.exclusions] == [
        ExclusionReason.ENDPOINT_COOLDOWN
    ]

    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger)
    result = _select(snapshot, registry, override_permit=permit, override_ledger=ledger)
    assert _providers(result) == ["test-first", "test-second"]
    assert result.exclusions == ()


def test_permit_is_scoped_to_its_own_endpoint(fake_clock: FakeClock) -> None:
    """A live permit bypasses only its endpoint's cooldown: the other
    endpoint's deployment stays excluded.
    """
    snapshot = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_deployment_cooldown(registry, "test-first")
    _arm_deployment_cooldown(registry, "test-second")

    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger, endpoint=OTHER_ENDPOINT_URL)
    assert permit.endpoint == OTHER_ENDPOINT_IDENTITY
    result = _select(snapshot, registry, override_permit=permit, override_ledger=ledger)
    # test-second's cooldown is bypassed; test-first's is not.
    assert _providers(result) == ["test-second"]
    assert [exclusion.reason for exclusion in result.exclusions] == [
        ExclusionReason.COOLDOWN
    ]
    assert result.exclusions[0].provider == "test-first"


def test_expired_permit_does_not_bypass(fake_clock: FakeClock) -> None:
    """A permit whose episode closed (measured recovery) is stale: the
    reservation survives the close, but the permit no longer bypasses.
    """
    snapshot = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_deployment_cooldown(registry)
    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger)
    assert ledger.note_endpoint_recovery(ENDPOINT_URL) is True
    assert ledger.permit_outstanding(permit) is False

    result = _select(snapshot, registry, override_permit=permit, override_ledger=ledger)
    assert _providers(result) == ["test-second"]
    assert [exclusion.reason for exclusion in result.exclusions] == [
        ExclusionReason.COOLDOWN
    ]


def test_used_permit_does_not_bypass(fake_clock: FakeClock) -> None:
    """A permit consumed at its transport start no longer bypasses."""
    snapshot = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_deployment_cooldown(registry)
    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger)
    assert ledger.start(permit).phase is OverridePhase.STARTED
    assert ledger.permit_outstanding(permit) is False

    result = _select(snapshot, registry, override_permit=permit, override_ledger=ledger)
    assert _providers(result) == ["test-second"]
    assert [exclusion.reason for exclusion in result.exclusions] == [
        ExclusionReason.COOLDOWN
    ]


def test_released_permit_does_not_bypass(fake_clock: FakeClock) -> None:
    """A pretransport-released permit no longer bypasses."""
    snapshot = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_deployment_cooldown(registry)
    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger)
    assert ledger.release(permit).phase is OverridePhase.RELEASED_PRETRANSPORT
    assert ledger.permit_outstanding(permit) is False

    result = _select(snapshot, registry, override_permit=permit, override_ledger=ledger)
    assert _providers(result) == ["test-second"]
    assert [exclusion.reason for exclusion in result.exclusions] == [
        ExclusionReason.COOLDOWN
    ]


def test_forged_permit_does_not_bypass(fake_clock: FakeClock) -> None:
    """A permit the ledger never minted (foreign token) never bypasses."""
    snapshot = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_deployment_cooldown(registry)
    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger)
    forged = replace(permit, token=permit.token + 100)
    assert ledger.permit_outstanding(forged) is False

    result = _select(snapshot, registry, override_permit=forged, override_ledger=ledger)
    assert _providers(result) == ["test-second"]
    assert [exclusion.reason for exclusion in result.exclusions] == [
        ExclusionReason.COOLDOWN
    ]


def test_permit_without_its_ledger_does_not_bypass(fake_clock: FakeClock) -> None:
    """Fail closed: a permit presented without the minting ledger never
    bypasses.
    """
    snapshot = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_deployment_cooldown(registry)
    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger)

    result = _select(snapshot, registry, override_permit=permit)
    assert _providers(result) == ["test-second"]
    assert [exclusion.reason for exclusion in result.exclusions] == [
        ExclusionReason.COOLDOWN
    ]


def test_disabled_deployment_is_still_excluded_with_a_permit(
    fake_clock: FakeClock,
) -> None:
    """The permit would bypass the armed cooldown, but the disabled check
    still excludes the deployment.
    """
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test-first": {"api_base": ENDPOINT_URL},
                "test-second": {"api_base": OTHER_ENDPOINT_URL},
            },
            "models": {
                "base": {
                    "deployments": [
                        {
                            "provider": "test-first",
                            "name": "first",
                            "supported_thinking_levels": ["off"],
                            "disabled": True,
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
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_deployment_cooldown(registry)
    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger)

    result = _select(snapshot, registry, override_permit=permit, override_ledger=ledger)
    assert _providers(result) == ["test-second"]
    assert [exclusion.reason for exclusion in result.exclusions] == [
        ExclusionReason.DISABLED
    ]
    assert result.exclusions[0].provider == "test-first"


def test_disabled_provider_is_still_excluded_with_a_permit(
    fake_clock: FakeClock,
) -> None:
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test-first": {"api_base": ENDPOINT_URL, "disabled": True},
                "test-second": {"api_base": OTHER_ENDPOINT_URL},
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
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_deployment_cooldown(registry)
    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger)

    result = _select(snapshot, registry, override_permit=permit, override_ledger=ledger)
    assert _providers(result) == ["test-second"]
    assert [exclusion.reason for exclusion in result.exclusions] == [
        ExclusionReason.DISABLED
    ]
    assert result.exclusions[0].provider == "test-first"


def test_allowlist_restriction_is_still_excluded_with_a_permit(
    fake_clock: FakeClock,
) -> None:
    snapshot = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_deployment_cooldown(registry)
    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger)

    with pytest.raises(AllDeploymentsUnavailableError) as exc_info:
        _select(
            snapshot,
            registry,
            allowed_models=["other-model"],
            override_permit=permit,
            override_ledger=ledger,
        )
    assert [exclusion.reason for exclusion in exc_info.value.exclusions] == [
        ExclusionReason.ALLOWLIST_EXCLUDED,
        ExclusionReason.ALLOWLIST_EXCLUDED,
    ]


def test_thinking_restriction_is_still_excluded_with_a_permit(
    fake_clock: FakeClock,
) -> None:
    """The cooldown bypass does not skip the compatibility check: a
    deployment that cannot serve the requested thinking level stays
    excluded even with a live matching permit.
    """
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test-first": {"api_base": ENDPOINT_URL},
                "test-second": {"api_base": OTHER_ENDPOINT_URL},
            },
            "models": {
                "base": {
                    "deployments": [
                        {
                            "provider": "test-first",
                            "name": "first",
                            "supported_thinking_levels": ["high"],
                        },
                        {
                            "provider": "test-second",
                            "name": "second",
                            "supported_thinking_levels": ["high"],
                        },
                    ]
                }
            },
            "roles": {},
        }),
        "test",
    )
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_deployment_cooldown(registry)
    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger)

    with pytest.raises(AllDeploymentsUnavailableError) as exc_info:
        _select(snapshot, registry, override_permit=permit, override_ledger=ledger)
    assert [exclusion.reason for exclusion in exc_info.value.exclusions] == [
        ExclusionReason.THINKING_UNSUPPORTED,
        ExclusionReason.THINKING_UNSUPPORTED,
    ]


def test_history_restriction_is_still_excluded_with_a_permit(
    fake_clock: FakeClock,
) -> None:
    snapshot = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_deployment_cooldown(registry)
    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger)
    image = ImageAttachment(
        source=InlineImageSource(data="image"), alias="image.png", mime_type="image/png"
    )
    history = [LLMMessage(role=Role.user, content="look", images=[image])]

    with pytest.raises(AllDeploymentsUnavailableError) as exc_info:
        _select(
            snapshot,
            registry,
            history=history,
            override_permit=permit,
            override_ledger=ledger,
        )
    assert [exclusion.reason for exclusion in exc_info.value.exclusions] == [
        ExclusionReason.IMAGES_UNSUPPORTED,
        ExclusionReason.IMAGES_UNSUPPORTED,
    ]


def test_compaction_restriction_is_still_excluded_with_a_permit(
    fake_clock: FakeClock,
) -> None:
    """A compaction base without a deployment on the candidate's provider
    stays excluded: the permit bypasses cooldowns, not compaction
    compatibility.
    """
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test-first": {"api_base": ENDPOINT_URL},
                "test-second": {"api_base": OTHER_ENDPOINT_URL},
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
                },
                "compact": {
                    "thinking": "off",
                    "deployments": [
                        {
                            "provider": "test-second",
                            "name": "compact",
                            "supported_thinking_levels": ["off"],
                        }
                    ],
                },
            },
            "roles": {},
        }),
        "test",
    )
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_deployment_cooldown(registry)
    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger)

    result = _select(
        snapshot,
        registry,
        compaction_base="compact",
        override_permit=permit,
        override_ledger=ledger,
    )
    # test-first has no compact deployment on its provider: excluded for
    # compaction incompatibility, not cooldown. test-second survives.
    assert _providers(result) == ["test-second"]
    assert [exclusion.reason for exclusion in result.exclusions] == [
        ExclusionReason.COMPACTION_INCOMPATIBLE
    ]
    assert result.exclusions[0].provider == "test-first"


def test_bypassed_selection_leaves_the_registry_untouched(
    fake_clock: FakeClock,
) -> None:
    """The bypass never mutates the registry: the cooldown stays armed, no
    probe is claimed or released for the bypassed candidate, and the permit
    stays outstanding for the transport-start consumption (C3).
    """
    snapshot = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_deployment_cooldown(registry)
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

    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger)
    result = _select(snapshot, registry, override_permit=permit, override_ledger=ledger)

    assert _providers(result) == ["test-first", "test-second"]
    # The cooldown stays armed: the bypass never clears registry state.
    assert registry.cooldown_until("base", "test-first") == 30.0
    assert registry.probe_in_flight("base", "test-first") is False
    # The bypassed candidate's admission is never consulted; the surviving
    # candidate's normal admission path is unchanged.
    assert claimed == [("base", "test-second")]
    assert released == []
    assert result.candidates[0].recovery_probe is False
    assert result.candidates[0].recovery_probe_token is None
    # The bypass does not consume the permit.
    assert ledger.permit_outstanding(permit) is True


def test_bypassed_selection_keeps_the_endpoint_cooldown_armed(
    fake_clock: FakeClock,
) -> None:
    snapshot = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_endpoint_cooldown(registry)
    until = registry.endpoint_cooldown_until(ENDPOINT_IDENTITY)
    assert until is not None

    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger)
    result = _select(snapshot, registry, override_permit=permit, override_ledger=ledger)

    assert _providers(result) == ["test-first", "test-second"]
    assert registry.endpoint_correlation_active(ENDPOINT_IDENTITY)
    assert registry.endpoint_cooldown_until(ENDPOINT_IDENTITY) == until


def test_read_only_eligibility_honors_the_permit_without_claiming(
    fake_clock: FakeClock,
) -> None:
    """The read-only view (``claim_probes=False``) honors the permit too,
    and still never claims or releases a probe.
    """
    snapshot = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    _arm_deployment_cooldown(registry)
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

    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger)
    result = _select(
        snapshot,
        registry,
        claim_probes=False,
        override_permit=permit,
        override_ledger=ledger,
    )

    assert _providers(result) == ["test-first", "test-second"]
    assert all(
        candidate.recovery_probe is False and candidate.recovery_probe_token is None
        for candidate in result.candidates
    )
    assert claimed == []
    assert released == []
    assert registry.cooldown_until("base", "test-first") == 30.0
    assert ledger.permit_outstanding(permit) is True


def _launch_config(thinking: ThinkingLevel = "off") -> ChartreuxConfigSchema:
    provider = ProviderConfig(
        name="provider", api_base=LAUNCH_ENDPOINT_URL, backend=Backend.GENERIC
    )
    return build_test_vibe_config(
        active_model="small",
        providers=[provider],
        models=[
            ModelConfig(
                name="small",
                alias="small",
                provider="provider",
                thinking=thinking,
                supported_thinking_levels=[thinking],
            )
        ],
    )


def _launch(
    schema: ChartreuxConfigSchema,
    parent: FakeConfigOrchestrator[ChartreuxConfigSchema],
    launch: LaunchConfig | None = None,
    **kwargs: Any,
) -> Any:
    kwargs.setdefault("authorized_tool_names", frozenset({"bash", "read_file"}))
    return resolve_launch(
        identity="coder",
        config=launch,
        parent_orchestrator=parent,
        tool_inventory={"bash": object(), "read_file": object()},
        **kwargs,
    )


def test_launch_assignment_bypasses_the_cooldown_for_a_matching_permit(
    fake_clock: FakeClock,
) -> None:
    config = _launch_config()
    parent: FakeConfigOrchestrator[ChartreuxConfigSchema] = FakeConfigOrchestrator(
        config
    )
    parent.availability_registry.record_failure(
        "small", "provider", FailureInfo(FailureCategory.TIMEOUT)
    )
    # Without a permit the cooled-down deployment cannot be assigned.
    with pytest.raises(InvalidLaunchModelError):
        _launch(config, parent)

    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger, endpoint=LAUNCH_ENDPOINT_URL)
    until = parent.availability_registry.cooldown_until("small", "provider")
    assert until is not None
    candidate = _launch(config, parent, override_permit=permit, override_ledger=ledger)
    assert candidate.effective_model.alias == "small"
    assert candidate.committed_model is not None
    assert candidate.committed_model.provider == "provider"
    # The bypass never mutates the registry: the cooldown stays armed.
    assert parent.availability_registry.cooldown_until("small", "provider") == until


def test_launch_assignment_rejects_wrong_expired_and_used_permits(
    fake_clock: FakeClock,
) -> None:
    config = _launch_config()

    def armed_parent() -> FakeConfigOrchestrator[ChartreuxConfigSchema]:
        parent = FakeConfigOrchestrator(config)
        parent.availability_registry.record_failure(
            "small", "provider", FailureInfo(FailureCategory.TIMEOUT)
        )
        return parent

    # Wrong endpoint: the permit is live but bound to another endpoint.
    ledger = CooldownOverrideLedger(clock=fake_clock)
    wrong = _reserved_permit(ledger, endpoint="https://other.test")
    with pytest.raises(InvalidLaunchModelError):
        _launch(config, armed_parent(), override_permit=wrong, override_ledger=ledger)

    # Expired: the episode closed (measured recovery) after the mint. The
    # stale reservation stays outstanding (it stays releasable), so each
    # scenario below draws on its own ledger.
    ledger = CooldownOverrideLedger(clock=fake_clock)
    expired = _reserved_permit(ledger, endpoint=LAUNCH_ENDPOINT_URL)
    assert ledger.note_endpoint_recovery(LAUNCH_ENDPOINT_URL) is True
    with pytest.raises(InvalidLaunchModelError):
        _launch(config, armed_parent(), override_permit=expired, override_ledger=ledger)

    # Used: the permit was consumed at its transport start.
    ledger = CooldownOverrideLedger(clock=fake_clock)
    used = _reserved_permit(ledger, endpoint=LAUNCH_ENDPOINT_URL)
    assert ledger.start(used).phase is OverridePhase.STARTED
    with pytest.raises(InvalidLaunchModelError):
        _launch(config, armed_parent(), override_permit=used, override_ledger=ledger)


def test_launch_compatibility_is_still_enforced_with_a_permit(
    fake_clock: FakeClock,
) -> None:
    """The permit bypasses the cooldown but not the thinking
    compatibility: an explicitly unsupported thinking level is still
    rejected at launch.
    """
    config = _launch_config(thinking="high")
    parent: FakeConfigOrchestrator[ChartreuxConfigSchema] = FakeConfigOrchestrator(
        config
    )
    parent.availability_registry.record_failure(
        "small", "provider", FailureInfo(FailureCategory.TIMEOUT)
    )
    ledger = CooldownOverrideLedger(clock=fake_clock)
    permit = _reserved_permit(ledger, endpoint=LAUNCH_ENDPOINT_URL)

    # The permit bypasses the cooldown: the default thinking is compatible.
    candidate = _launch(config, parent, override_permit=permit, override_ledger=ledger)
    assert candidate.effective_model.alias == "small"
    # ...but an explicitly unsupported thinking level is still rejected.
    with pytest.raises(InvalidLaunchModelError):
        _launch(
            config,
            parent,
            launch=LaunchConfig(thinking="off"),
            override_permit=permit,
            override_ledger=ledger,
        )
