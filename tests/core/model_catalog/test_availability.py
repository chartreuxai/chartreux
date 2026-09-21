from __future__ import annotations

from typing import Any, cast

import pytest

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
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
from chartreux.core.model_catalog.resolver import ModelResolver
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.session_types import CommittedModelIdentity
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


def test_resolver_skips_incompatible_or_cooled_tag_members() -> None:
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test/first": {"api_base": "https://first.test"},
                "test/second": {"api_base": "https://second.test"},
            },
            "models": {
                "incompatible": {
                    "deployments": [{"provider": "test/first", "name": "first"}]
                },
                "available": {
                    "deployments": [{"provider": "test/second", "name": "second"}]
                },
            },
            "tags": {"preferred": ["incompatible", "available"]},
        }),
        "selection",
    )
    resolver = ModelResolver(snapshot)
    accepted = lambda candidate: candidate.base_model == "available"
    assert (
        resolver.resolve("@preferred", candidate_filter=accepted).base_model
        == "available"
    )


def test_compaction_checks_actual_destination_deployment_capabilities() -> None:
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {"test/first": {"api_base": "https://first.test"}},
            "models": {
                "base": {
                    "deployments": [
                        {
                            "provider": "test/first",
                            "name": "base",
                            "supports_images": True,
                        }
                    ]
                },
                "compact": {
                    "deployments": [
                        {
                            "provider": "test/first",
                            "name": "compact",
                            "supports_images": False,
                        }
                    ]
                },
            },
            "tags": {},
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
                provider="test/first",
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
            "providers": {"test/first": {"api_base": "https://first.test"}},
            "models": {
                "base": {"deployments": [{"provider": "test/first", "name": "base"}]},
                "compact": {
                    "deployments": [
                        {
                            "provider": "test/first",
                            "name": "compact",
                            "supported_thinking_levels": ["off"],
                        }
                    ]
                },
            },
            "tags": {},
        }),
        "compaction",
    )
    config = _config(snapshot, thinking_overrides={"compact": "high"})

    with pytest.raises(AllDeploymentsUnavailableError) as error:
        eligible_deployments(
            snapshot=snapshot,
            committed=CommittedModelIdentity(
                base_model="base",
                provider="test/first",
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


def test_compaction_alias_uses_its_canonical_base_for_eligibility() -> None:
    raw = _catalog().catalog.model_dump()
    raw["models"]["compact"]["aliases"] = ["short"]  # type: ignore[index]
    raw["models"]["compact"]["deployments"] = [  # type: ignore[index]
        {"provider": "test/first", "name": "compact-first"},
        {"provider": "test/second", "name": "compact-second"},
    ]
    snapshot = CatalogSnapshot(ModelCatalog.model_validate(raw), "alias")

    result = eligible_deployments(
        snapshot=snapshot,
        committed=_committed(),
        registry=AvailabilityRegistry(),
        config=_config(snapshot),
        thinking="off",
        compaction_base="short",
    )

    assert [item.resolved.deployment.provider for item in result.candidates] == [
        "test/first",
        "test/second",
    ]
    assert ExclusionReason.COMPACTION_INCOMPATIBLE not in {
        item.reason for item in result.exclusions
    }


def test_compaction_alias_without_canonical_deployment_excludes_provider() -> None:
    raw = _catalog().catalog.model_dump()
    raw["models"]["compact"]["aliases"] = ["short"]  # type: ignore[index]
    snapshot = CatalogSnapshot(ModelCatalog.model_validate(raw), "alias")

    result = eligible_deployments(
        snapshot=snapshot,
        committed=_committed(),
        registry=AvailabilityRegistry(),
        config=_config(snapshot),
        thinking="off",
        compaction_base="short",
    )

    assert [item.resolved.deployment.provider for item in result.candidates] == [
        "test/second"
    ]
    assert ("test/first", ExclusionReason.COMPACTION_INCOMPATIBLE) in [
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
    assert registry.admission("base", "provider")[:2] == (True, True)
    registry.record_success("base", "provider")
    assert registry.cooldown_until("base", "provider") is None
    registry.record_failure("base", "provider")
    now[0] = 4.0
    assert registry.admission("base", "provider")[:2] == (True, True)
    registry.record_failure("base", "provider")
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

    registry.record_failure("base", "provider")
    now[0] = 4.0
    _, _, newer_token = registry.admission("base", "provider")
    assert newer_token is not None
    assert newer_token != first_token

    registry.release_probe("base", "provider", first_token)
    assert registry.admission("base", "provider")[:2] == (False, False)
    registry.release_probe("base", "provider", newer_token)
    assert registry.admission("base", "provider")[:2] == (True, True)


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
                "test/first": {"api_base": "https://first.test"},
                "test/second": {"api_base": "https://second.test"},
                "test/disabled": {
                    "api_base": "https://disabled.test",
                    "disabled": True,
                },
            },
            "models": {
                "base": {
                    "deployments": [
                        {
                            "provider": "test/first",
                            "name": "first",
                            "supported_thinking_levels": ["off"],
                        },
                        {
                            "provider": "test/second",
                            "name": "second",
                            "supported_thinking_levels": ["off"],
                        },
                        {"provider": "test/disabled", "name": "disabled"},
                    ]
                },
                "compact": {
                    "deployments": [
                        {"provider": "test/second", "name": "compact-second"}
                    ]
                },
            },
            "tags": {},
        }),
        "test",
    )


def _config(snapshot: CatalogSnapshot, **updates: object) -> ChartreuxConfigSchema:
    return ChartreuxConfigSchema(**cast(Any, updates)).attach_catalog_snapshot(snapshot)


def _committed() -> CommittedModelIdentity:
    return CommittedModelIdentity(
        base_model="base",
        provider="test/first",
        wire_name="first",
        catalog_revision="test",
    )


def test_all_unavailable_names_base_and_returns_structured_exclusions() -> None:
    snapshot = _catalog()
    registry = AvailabilityRegistry()
    registry.record_failure("base", "test/first")
    registry.record_failure("base", "test/second")
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
    registry.record_failure("base", "test/first")
    result = eligible_deployments(
        snapshot=snapshot,
        committed=_committed(),
        registry=registry,
        config=_config(snapshot),
        thinking="off",
    )
    assert [item.resolved.deployment.provider for item in result.candidates] == [
        "test/second"
    ]
    assert [(item.provider, item.reason) for item in result.exclusions] == [
        ("test/first", ExclusionReason.COOLDOWN),
        ("test/disabled", ExclusionReason.DISABLED),
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
        "test/second"
    ]
    assert [(item.provider, item.reason) for item in result.exclusions] == [
        ("test/first", ExclusionReason.COMPACTION_INCOMPATIBLE),
        ("test/disabled", ExclusionReason.DISABLED),
    ]
