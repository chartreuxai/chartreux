from __future__ import annotations

from typing import Any, cast

import pytest

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers import DefaultConfigLayer, OverridesLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.llm.failures import FailureCategory, FailureInfo
from chartreux.core.llm_models import (
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
from chartreux.core.model_catalog.resolver import ModelResolutionError, ModelResolver
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.session_types import CommittedModelIdentity
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


def test_resolver_does_not_substitute_another_model_for_role_preset() -> None:
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test-first": {"api_base": "https://first.test"},
                "test-second": {"api_base": "https://second.test"},
            },
            "models": {
                "incompatible": {
                    "deployments": [{"provider": "test-first", "name": "first"}]
                },
                "available": {
                    "deployments": [{"provider": "test-second", "name": "second"}]
                },
            },
            "roles": {
                "preferred": {
                    "description": "Preferred preset",
                    "model": "incompatible",
                    "thinking": "high",
                }
            },
        }),
        "selection",
    )
    resolver = ModelResolver(snapshot)
    accepted = lambda candidate: candidate.base_model == "available"
    with pytest.raises(ModelResolutionError, match="Model 'incompatible'") as error:
        resolver.resolve("@preferred", candidate_filter=accepted)
    assert error.value.code == "no_compatible_deployment"


def test_compaction_checks_actual_destination_deployment_capabilities() -> None:
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {"test-first": {"api_base": "https://first.test"}},
            "models": {
                "base": {
                    "deployments": [
                        {
                            "provider": "test-first",
                            "name": "base",
                            "supports_images": True,
                        }
                    ]
                },
                "compact": {
                    "deployments": [
                        {
                            "provider": "test-first",
                            "name": "compact",
                            "supports_images": False,
                        }
                    ]
                },
            },
            "roles": {},
        }),
        "compaction",
    )
    history = [
        LLMMessage(
            role=Role.user,
            images=[
                ImageAttachment(
                    source=InlineImageSource(data="image"),
                    alias="image.png",
                    mime_type="image/png",
                )
            ],
        )
    ]
    with pytest.raises(AllDeploymentsUnavailableError) as error:
        eligible_deployments(
            snapshot=snapshot,
            committed=CommittedModelIdentity(
                base_model="base",
                provider="test-first",
                wire_name="base",
                catalog_revision="compaction",
            ),
            registry=AvailabilityRegistry(),
            config=_config(snapshot),
            history=history,
            thinking="off",
            thinking_explicit=False,
            compaction_base="compact",
        )
    assert [item.reason for item in error.value.exclusions] == [
        ExclusionReason.COMPACTION_INCOMPATIBLE
    ]


def test_compaction_destination_declared_thinking_restriction_excludes_requested_level() -> (
    None
):
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {"test-first": {"api_base": "https://first.test"}},
            "models": {
                "base": {"deployments": [{"provider": "test-first", "name": "base"}]},
                "compact": {
                    "deployments": [
                        {
                            "provider": "test-first",
                            "name": "compact",
                            "supported_thinking_levels": ["off"],
                        }
                    ]
                },
            },
            "roles": {},
        }),
        "compaction",
    )
    config = _config(snapshot, thinking_overrides={"compact": "high"})

    with pytest.raises(AllDeploymentsUnavailableError) as error:
        eligible_deployments(
            snapshot=snapshot,
            committed=CommittedModelIdentity(
                base_model="base",
                provider="test-first",
                wire_name="base",
                catalog_revision="compaction",
            ),
            registry=AvailabilityRegistry(),
            config=config,
            thinking="off",
            thinking_explicit=False,
            compaction_base="compact",
        )

    assert [item.reason for item in error.value.exclusions] == [
        ExclusionReason.COMPACTION_INCOMPATIBLE
    ]


def test_compaction_canonical_base_for_eligibility() -> None:
    raw = _catalog().catalog.model_dump()
    raw["models"]["compact"]["deployments"] = [  # type: ignore[index]
        {"provider": "test-first", "name": "compact-first"},
        {"provider": "test-second", "name": "compact-second"},
    ]
    snapshot = CatalogSnapshot(ModelCatalog.model_validate(raw), "alias")

    result = eligible_deployments(
        snapshot=snapshot,
        committed=_committed(),
        registry=AvailabilityRegistry(),
        config=_config(snapshot),
        thinking="off",
        compaction_base="compact",
    )

    assert [item.resolved.deployment.provider for item in result.candidates] == [
        "test-first",
        "test-second",
    ]
    assert ExclusionReason.COMPACTION_INCOMPATIBLE not in {
        item.reason for item in result.exclusions
    }


def test_compaction_without_canonical_deployment_excludes_provider() -> None:
    raw = _catalog().catalog.model_dump()
    snapshot = CatalogSnapshot(ModelCatalog.model_validate(raw), "alias")

    result = eligible_deployments(
        snapshot=snapshot,
        committed=_committed(),
        registry=AvailabilityRegistry(),
        config=_config(snapshot),
        thinking="off",
        compaction_base="compact",
    )

    assert [item.resolved.deployment.provider for item in result.candidates] == [
        "test-second"
    ]
    assert ("test-first", ExclusionReason.COMPACTION_INCOMPATIBLE) in [
        (item.provider, item.reason) for item in result.exclusions
    ]


def test_root_and_child_orchestrators_share_one_availability_registry() -> None:
    root = FakeConfigOrchestrator(_config(_catalog()))
    child = root._copy_for_child()
    assert child.availability_registry is root.availability_registry

    now = [100.0]
    registry = AvailabilityRegistry(clock=lambda: now[0])
    registry.record_failure("base", "p1")
    assert registry.cooldown_until("base", "p1") == pytest.approx(130.0)
    registry.record_failure(
        "base", "p2", FailureInfo(FailureCategory.RATE_LIMIT, retry_after=75.0)
    )
    assert registry.cooldown_until("base", "p2") == pytest.approx(175.0)


@pytest.mark.asyncio
async def test_real_root_child_and_sibling_filter_shared_cooldown_and_probe() -> None:
    snapshot = _catalog()
    overrides = OverridesLayer(data={"active_model": "base"})
    root = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), overrides],
        default_layer_resolver=lambda: overrides,
        catalog_snapshot=snapshot,
    )
    now = [0.0]
    registry = AvailabilityRegistry(initial_cooldown=1.0, clock=lambda: now[0])
    root._availability_registry = registry
    child = root._copy_for_child()
    sibling = root._copy_for_child()
    child.availability_registry.record_failure("base", "test-first")

    for runtime in (root, child, sibling):
        assert runtime.availability_registry is registry
        result = eligible_deployments(
            snapshot=snapshot,
            committed=_committed(),
            registry=runtime.availability_registry,
            config=runtime.config,
            thinking="off",
        )
        assert [item.resolved.deployment.provider for item in result.candidates] == [
            "test-second"
        ]
        assert result.exclusions[0].reason is ExclusionReason.COOLDOWN

    now[0] = 2.0
    admitted = eligible_deployments(
        snapshot=snapshot,
        committed=_committed(),
        registry=child.availability_registry,
        config=child.config,
        thinking="off",
    ).candidates[0]
    assert admitted.recovery_probe
    assert admitted.recovery_probe_token is not None
    assert not sibling.availability_registry.is_available("base", "test-first")
    assert registry.admission("base", "test-first")[:2] == (False, False)
    registry.release_probe("base", "test-first", admitted.recovery_probe_token)
    assert root.availability_registry.is_available("base", "test-first")


def test_expiry_admits_exactly_one_probe_per_key() -> None:
    now = [0.0]
    registry = AvailabilityRegistry(initial_cooldown=1.0, clock=lambda: now[0])
    registry.record_failure("base", "p1")
    registry.record_failure("base", "p2")
    now[0] = 2.0
    assert registry.admission("base", "p1")[:2] == (True, True)
    assert registry.admission("base", "p1")[:2] == (False, False)
    assert registry.admission("base", "p2")[:2] == (True, True)


def test_probe_success_clears_and_failure_extends() -> None:
    now = [0.0]
    registry = AvailabilityRegistry(initial_cooldown=1.0, clock=lambda: now[0])
    registry.record_failure("base", "provider")
    now[0] = 2.0
    admitted, is_probe, token = registry.admission("base", "provider")
    assert (admitted, is_probe) == (True, True)
    assert token is not None
    # Review R13: the settlement is token-owned — the probe OWNER's
    # success clears the cooldown (and releases the claim).
    registry.record_success("base", "provider", probe_token=token)
    assert registry.cooldown_until("base", "provider") is None
    registry.record_failure("base", "provider")
    now[0] = 4.0
    admitted, is_probe, token = registry.admission("base", "provider")
    assert (admitted, is_probe) == (True, True)
    assert token is not None
    # The owner's failure extends the cooldown (and releases the claim).
    registry.record_failure("base", "provider", probe_token=token)
    assert registry.cooldown_until("base", "provider") == pytest.approx(5.0)


def test_release_unattempted_probe_permits_another_claim() -> None:
    now = [0.0]
    registry = AvailabilityRegistry(initial_cooldown=1.0, clock=lambda: now[0])
    registry.record_failure("base", "provider")
    now[0] = 2.0
    admitted, is_probe, token = registry.admission("base", "provider")
    assert (admitted, is_probe) == (True, True)
    assert token is not None
    registry.release_probe("base", "provider", token)
    assert registry.admission("base", "provider")[:2] == (True, True)


def test_stale_probe_release_cannot_release_a_newer_claim() -> None:
    now = [0.0]
    registry = AvailabilityRegistry(initial_cooldown=1.0, clock=lambda: now[0])
    registry.record_failure("base", "provider")
    now[0] = 2.0
    _, _, first_token = registry.admission("base", "provider")
    assert first_token is not None

    # Review R13: the settlement is token-owned — the first probe's OWNER
    # settles it (re-arming the cooldown), which alone makes the key
    # claimable again for the newer probe.
    registry.record_failure("base", "provider", probe_token=first_token)
    now[0] = 4.0
    _, _, newer_token = registry.admission("base", "provider")
    assert newer_token is not None
    assert newer_token != first_token

    registry.release_probe("base", "provider", first_token)
    assert registry.admission("base", "provider")[:2] == (False, False)
    registry.release_probe("base", "provider", newer_token)
    assert registry.admission("base", "provider")[:2] == (True, True)


def test_out_of_band_error_mid_probe_leaves_no_stranded_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-2 V3 (a): an exception between claiming a recovery probe and
    handing the candidate to the caller must not strand the claim.

    FAILS pre-fix: eligible_deployments claimed the probe for the first
    deployment and the compatibility check then raised — probe_in_flight
    stayed True, so the deployment stayed probe-blocked until a user
    reset_base.
    """
    snapshot = _catalog()
    now = [0.0]
    registry = AvailabilityRegistry(initial_cooldown=1.0, clock=lambda: now[0])
    registry.record_failure("base", "test-first")
    now[0] = 2.0  # cooldown expired: the eligibility pass claims the probe

    def boom(**kwargs: object) -> None:
        raise RuntimeError("out-of-band probe error")

    monkeypatch.setattr(
        "chartreux.core.model_catalog.availability.compatibility_exclusion", boom
    )
    with pytest.raises(RuntimeError, match="out-of-band probe error"):
        eligible_deployments(
            snapshot=snapshot,
            committed=_committed(),
            registry=registry,
            config=_config(snapshot),
            thinking="off",
        )
    # The claim was released on the way out: nothing is in flight and the
    # deployment is claimable again without a user reset_base.
    assert not registry.probe_in_flight("base", "test-first")
    assert registry.admission("base", "test-first")[:2] == (True, True)


def test_out_of_band_error_releases_earlier_candidates_claims_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-2 V3 (a), multi-deployment: an exception on a LATER deployment
    must also release the probes already claimed for EARLIER candidates —
    the caller never receives them when eligible_deployments raises.

    FAILS pre-fix: both deployments stayed probe-blocked (stranded claims).
    """
    snapshot = _catalog()
    now = [0.0]
    registry = AvailabilityRegistry(initial_cooldown=1.0, clock=lambda: now[0])
    registry.record_failure("base", "test-first")
    registry.record_failure("base", "test-second")
    now[0] = 2.0  # both cooldowns expired: the pass claims both probes

    calls = [0]

    def boom_on_second(**kwargs: object) -> None:
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("out-of-band probe error")
        return None

    monkeypatch.setattr(
        "chartreux.core.model_catalog.availability.compatibility_exclusion",
        boom_on_second,
    )
    with pytest.raises(RuntimeError, match="out-of-band probe error"):
        eligible_deployments(
            snapshot=snapshot,
            committed=_committed(),
            registry=registry,
            config=_config(snapshot),
            thinking="off",
        )
    assert not registry.probe_in_flight("base", "test-first")
    assert not registry.probe_in_flight("base", "test-second")
    assert registry.admission("base", "test-first")[:2] == (True, True)
    assert registry.admission("base", "test-second")[:2] == (True, True)


def test_stale_probe_claim_stops_blocking_after_the_ttl() -> None:
    """Round-2 V3 (b): a probe claim carries a wall-clock TTL — a claim older
    than the bound is dead (no live probe attempt can outlive the request
    timeout plus the retry budget), so it stops blocking availability and
    can be re-claimed under a NEW owner token.

    FAILS pre-fix: the constructor has no TTL (TypeError) and a claim never
    expires — the deployment stays probe-blocked until a user reset_base.
    """
    now = [0.0]
    registry = AvailabilityRegistry(
        initial_cooldown=1.0, clock=lambda: now[0], probe_claim_ttl=10.0
    )
    registry.record_failure("base", "provider")
    now[0] = 2.0  # cooldown expired
    admitted, is_probe, first_token = registry.admission("base", "provider")
    assert (admitted, is_probe) == (True, True)
    assert first_token is not None
    assert registry.probe_in_flight("base", "provider")
    assert not registry.is_available("base", "provider")

    now[0] = 13.0  # beyond the 10s claim TTL: the claim is dead
    assert registry.is_available("base", "provider")
    assert not registry.probe_in_flight("base", "provider")
    admitted, is_probe, second_token = registry.admission("base", "provider")
    assert (admitted, is_probe) == (True, True)
    assert second_token is not None
    assert second_token != first_token
    # The re-claim is fresh: it blocks again until its own owner settles it.
    assert registry.probe_in_flight("base", "provider")
    assert not registry.is_available("base", "provider")


def test_stale_probe_claim_never_lifts_an_active_cooldown() -> None:
    """Round-3 W4: staleness never lifts a cooldown.

    ``is_available``/``admission`` evaluate the cooldown's ``until`` BEFORE
    the claim's staleness, so a STALE probe claim combined with an ACTIVE
    unexpired cooldown still reports unavailable — pin the evaluation order
    so a future refactor cannot flip it (the TTL only unblocks a claim whose
    cooldown has already expired).
    """
    now = [0.0]
    registry = AvailabilityRegistry(
        initial_cooldown=1.0, clock=lambda: now[0], probe_claim_ttl=10.0
    )
    registry.record_failure("base", "provider")
    now[0] = 2.0  # cooldown expired: the probe claim is taken
    admitted, is_probe, token = registry.admission("base", "provider")
    assert (admitted, is_probe) == (True, True)
    assert token is not None
    # A late normal attempt (no owner token) extends the cooldown while the
    # probe claim stays outstanding.
    registry.record_failure(
        "base", "provider", FailureInfo(FailureCategory.RATE_LIMIT, retry_after=100.0)
    )
    now[0] = 13.0  # claim age 11s > the 10s TTL: the claim is STALE...
    # ...but the cooldown is still active: staleness never lifts it.
    assert not registry.is_available("base", "provider")
    assert registry.admission("base", "provider") == (False, False, None)
    # Once the cooldown expires, the stale claim no longer blocks either.
    now[0] = 103.0
    assert registry.is_available("base", "provider")
    admitted, is_probe, reclaimed = registry.admission("base", "provider")
    assert (admitted, is_probe) == (True, True)
    assert reclaimed is not None
    assert reclaimed != token


def test_orphaned_token_never_releases_a_reclaimed_probe() -> None:
    """Round-2 V3 (b): token-ownership survives a TTL re-claim — the
    orphaned claim's late release is a no-op and never touches the new
    owner's claim (release stays token-owned).

    FAILS pre-fix: the constructor has no TTL (TypeError).
    """
    now = [0.0]
    registry = AvailabilityRegistry(
        initial_cooldown=1.0, clock=lambda: now[0], probe_claim_ttl=10.0
    )
    registry.record_failure("base", "provider")
    now[0] = 2.0
    _, _, first_token = registry.admission("base", "provider")
    assert first_token is not None
    now[0] = 13.0  # first claim is stale; re-claim under a new token
    _, _, second_token = registry.admission("base", "provider")
    assert second_token is not None
    assert second_token != first_token

    # The orphaned owner's late release must not release the new claim.
    registry.release_probe("base", "provider", first_token)
    assert registry.probe_in_flight("base", "provider")
    assert registry.admission("base", "provider")[:2] == (False, False)
    # The new owner's release settles it.
    registry.release_probe("base", "provider", second_token)
    assert not registry.probe_in_flight("base", "provider")
    assert registry.admission("base", "provider")[:2] == (True, True)


def test_eligible_deployments_reclaims_a_stale_probe_claim() -> None:
    """Round-2 V3 (b), integration: a stranded (abandoned) probe claim past
    its TTL no longer blocks the eligibility pass — the deployment is
    admitted with a fresh owner token.

    FAILS pre-fix: the constructor has no TTL (TypeError); without a TTL
    the abandoned claim probe-blocks the deployment (the first candidate
    would be test-second, not test-first).
    """
    snapshot = _catalog()
    now = [0.0]
    registry = AvailabilityRegistry(
        initial_cooldown=1.0, clock=lambda: now[0], probe_claim_ttl=10.0
    )
    registry.record_failure("base", "test-first")
    now[0] = 2.0
    # A probe claimed and abandoned (the stranded-claim shape V3 fixes at
    # the source; the TTL is the backstop).
    _, _, stranded_token = registry.admission("base", "test-first")
    assert stranded_token is not None
    now[0] = 13.0  # the abandoned claim is now stale
    result = eligible_deployments(
        snapshot=snapshot,
        committed=_committed(),
        registry=registry,
        config=_config(snapshot),
        thinking="off",
    )
    candidate = result.candidates[0]
    assert candidate.resolved.deployment.provider == "test-first"
    assert candidate.recovery_probe
    assert candidate.recovery_probe_token is not None
    assert candidate.recovery_probe_token != stranded_token


def test_keys_are_independent() -> None:
    now = [0.0]
    registry = AvailabilityRegistry(initial_cooldown=10.0, clock=lambda: now[0])
    registry.record_failure("base-a", "provider")
    assert registry.admission("base-a", "provider")[:2] == (False, False)
    assert registry.admission("base-b", "provider")[:2] == (True, False)
    assert registry.admission("base-a", "other")[:2] == (True, False)


def _catalog() -> CatalogSnapshot:
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test-first": {"api_base": "https://first.test"},
                "test-second": {"api_base": "https://second.test"},
                "test-disabled": {
                    "api_base": "https://disabled.test",
                    "disabled": True,
                },
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
                        {"provider": "test-disabled", "name": "disabled"},
                    ]
                },
                "compact": {
                    "deployments": [
                        {"provider": "test-second", "name": "compact-second"}
                    ]
                },
            },
            "roles": {},
        }),
        "test",
    )


def _config(snapshot: CatalogSnapshot, **updates: object) -> ChartreuxConfigSchema:
    return ChartreuxConfigSchema(
        active_model="base", **cast(Any, updates)
    ).attach_catalog_snapshot(snapshot)


def _committed() -> CommittedModelIdentity:
    return CommittedModelIdentity(
        base_model="base",
        provider="test-first",
        wire_name="first",
        catalog_revision="test",
    )


def test_all_unavailable_names_base_and_returns_structured_exclusions() -> None:
    snapshot = _catalog()
    registry = AvailabilityRegistry()
    registry.record_failure("base", "test-first")
    registry.record_failure("base", "test-second")
    with pytest.raises(AllDeploymentsUnavailableError) as error:
        eligible_deployments(
            snapshot=snapshot,
            committed=_committed(),
            registry=registry,
            config=_config(snapshot),
            thinking="off",
        )
    assert error.value.base_model == "base"
    assert {item.reason for item in error.value.exclusions} == {
        ExclusionReason.COOLDOWN,
        ExclusionReason.DISABLED,
    }


def test_candidate_priority_cooldown_filtering_and_disabled_skip_are_data() -> None:
    snapshot = _catalog()
    registry = AvailabilityRegistry()
    registry.record_failure("base", "test-first")
    result = eligible_deployments(
        snapshot=snapshot,
        committed=_committed(),
        registry=registry,
        config=_config(snapshot),
        thinking="off",
    )
    assert [item.resolved.deployment.provider for item in result.candidates] == [
        "test-second"
    ]
    assert [(item.provider, item.reason) for item in result.exclusions] == [
        ("test-first", ExclusionReason.COOLDOWN),
        ("test-disabled", ExclusionReason.DISABLED),
    ]


def test_revalidation_exclusions_preserve_history_and_thinking() -> None:
    snapshot = _catalog()
    config = _config(snapshot)
    history = [
        LLMMessage(
            role=Role.user,
            content="image",
            images=[
                ImageAttachment(
                    source=InlineImageSource(data="image"),
                    alias="image.png",
                    mime_type="image/png",
                )
            ],
        )
    ]
    with pytest.raises(AllDeploymentsUnavailableError) as error:
        eligible_deployments(
            snapshot=snapshot,
            committed=_committed(),
            registry=AvailabilityRegistry(),
            config=config,
            history=history,
            thinking="max",
        )
    assert history[0].images is not None
    assert len(history[0].images) == 1
    assert ExclusionReason.IMAGES_UNSUPPORTED in {
        item.reason for item in error.value.exclusions
    }


@pytest.mark.parametrize(
    ("history", "thinking", "reason"),
    [
        pytest.param(
            [
                LLMMessage(
                    role=Role.user,
                    content="image",
                    images=[
                        ImageAttachment(
                            source=InlineImageSource(data="image"),
                            alias="image.png",
                            mime_type="image/png",
                        )
                    ],
                )
            ],
            "off",
            ExclusionReason.IMAGES_UNSUPPORTED,
            id="images",
        ),
        pytest.param(
            [LLMMessage(role=Role.assistant, reasoning_payloads=[{"type": "opaque"}])],
            "off",
            ExclusionReason.REASONING_UNSUPPORTED,
            id="reasoning",
        ),
        pytest.param([], "max", ExclusionReason.THINKING_UNSUPPORTED, id="thinking"),
    ],
)
def test_revalidation_matrix_never_strips_history_or_lowers_thinking(
    history: list[LLMMessage], thinking: str, reason: ExclusionReason
) -> None:
    snapshot = _catalog()
    before = [message.model_dump(mode="json") for message in history]
    with pytest.raises(AllDeploymentsUnavailableError) as error:
        eligible_deployments(
            snapshot=snapshot,
            committed=_committed(),
            registry=AvailabilityRegistry(),
            config=_config(snapshot),
            history=history,
            thinking=cast(Any, thinking),
        )
    assert reason in {item.reason for item in error.value.exclusions}
    assert [message.model_dump(mode="json") for message in history] == before
    assert (
        thinking
        == {"images": "off", "reasoning": "off", "thinking": "max"}[
            reason.value.split("-")[0]
        ]
    )
    snapshot = _catalog()
    result = eligible_deployments(
        snapshot=snapshot,
        committed=_committed(),
        registry=AvailabilityRegistry(),
        config=_config(snapshot),
        thinking="off",
        compaction_base="compact",
    )
    assert [item.resolved.deployment.provider for item in result.candidates] == [
        "test-second"
    ]
    assert [(item.provider, item.reason) for item in result.exclusions] == [
        ("test-first", ExclusionReason.COMPACTION_INCOMPATIBLE),
        ("test-disabled", ExclusionReason.DISABLED),
    ]
