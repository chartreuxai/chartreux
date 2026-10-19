"""WP1b — validated typed recovery policy config (plan state.md WP1).

Typed config declares counts/delays/jitter/correlation thresholds/deadlines/
eligible classes/suspension limits/recovery concurrency/eligible fallback
constraints. ``eligible_classes`` is the ONE authoritative recovery-layer
allowlist (state.md R0 eligibility note): the transient FailureInfo classes
connection/timeout/server_error/overload/rate_limit, the WP1 typed transient
dispositions incomplete_stream/empty_response, and the transient form of
launch_failure. The low-level classes conflict/too_early are deliberately NOT
recovery-eligible — decided explicitly here, by exclusion.

WP4 (round-3 symmetry amendment): the child controller's policy is the
``[recovery.subagents]`` subsection of the ``[recovery]`` container
(``RecoverySectionConfig``); ``[recovery.root]`` (R2) joins it symmetrically.

AUTOMATIC RECOVERY IS ENABLED BY DEFAULT since v0.5.6: ``enabled`` defaults
to True (promoted at WP6 per OD4b/OD5 once WP4's acceptance matrix passed).
Strict validation rejects negative counts, NaN/infinite durations, unknown
classes, and contradictory constraints. Every deadline is an ADVISORY
scheduling/admission bound (Q2/F3: no hard in-flight kill exists — the retry
budget is consulted only between attempts).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest

from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.models import (
    RecoveryConfig,
    RecoveryEligibleClass,
    RecoverySectionConfig,
)
from chartreux.core.config.orchestrator import _availability_registry_from_config
from chartreux.core.model_catalog.availability import AvailabilityRegistry

AUTHORITATIVE_ALLOWLIST = [
    "connection",
    "timeout",
    "server_error",
    "overload",
    "rate_limit",
    "launch_failure",
    "incomplete_stream",
    "empty_response",
]


def test_automatic_recovery_is_enabled_by_default() -> None:
    # WP6 default promotion (OD5): the shipped default is enabled.
    assert RecoveryConfig().enabled is True
    assert ChartreuxConfigSchema().recovery.subagents.enabled is True
    # The container's default is a fully default (enabled) subagent policy.
    assert ChartreuxConfigSchema().recovery.subagents == RecoveryConfig()


def test_default_policy_matches_handoff_initial_config() -> None:
    config = RecoveryConfig()

    assert config.auto_retries_max == 2
    assert config.same_agent_resumes_max == 2
    assert config.cooldowns_seconds == [30.0, 120.0]
    assert config.jitter_fraction == 0.2
    assert config.episode_deadline_seconds == 2400.0
    assert config.endpoint_correlation_window_seconds == 300.0
    assert config.endpoint_correlation_min_events == 2
    assert config.endpoint_correlation_min_models == 2
    assert config.endpoint_cooldown_seconds == 30.0
    assert config.suspension_ttl_seconds == 1800.0
    assert config.max_suspended_agents == 8
    assert config.recovery_concurrency == 2
    assert config.fallback_require_endpoint_independence is True
    assert config.fallback_allowed_base_models == []
    assert config.fallback_max_output_price_per_million is None


def test_eligible_classes_default_to_the_authoritative_allowlist() -> None:
    assert [member.value for member in RecoveryEligibleClass] == (
        AUTHORITATIVE_ALLOWLIST
    )
    assert [member.value for member in RecoveryConfig().eligible_classes] == (
        AUTHORITATIVE_ALLOWLIST
    )


@pytest.mark.parametrize("excluded", ["conflict", "too_early"])
def test_low_level_conflict_and_too_early_are_not_recovery_eligible(
    excluded: str,
) -> None:
    # Deliberate WP1b decision (state.md R0 eligibility note): these stay
    # transport-retry-eligible only; recovery-layer eligibility is decided
    # explicitly, by exclusion from the authoritative allowlist.
    with pytest.raises(ValueError):
        RecoveryEligibleClass(excluded)
    with pytest.raises(ValidationError, match="eligible_classes"):
        RecoveryConfig.model_validate({"eligible_classes": [excluded]})


def test_eligible_classes_accept_enum_members_and_string_subsets() -> None:
    config = RecoveryConfig(eligible_classes=[RecoveryEligibleClass.TIMEOUT])

    assert config.eligible_classes == [RecoveryEligibleClass.TIMEOUT]

    coerced = RecoveryConfig.model_validate({
        "eligible_classes": ["timeout", "incomplete_stream"]
    })

    assert coerced.eligible_classes == [
        RecoveryEligibleClass.TIMEOUT,
        RecoveryEligibleClass.INCOMPLETE_STREAM,
    ]


def test_schema_recovery_section_sparse_dict_keeps_typed_defaults() -> None:
    assert "recovery" in ChartreuxConfigSchema.model_fields

    config = ChartreuxConfigSchema.model_validate({
        "recovery": {"subagents": {"enabled": True}}
    })

    subagents = config.recovery.subagents
    assert subagents.enabled is True
    assert subagents.auto_retries_max == 2
    assert subagents.cooldowns_seconds == [30.0, 120.0]
    assert subagents.jitter_fraction == 0.2
    assert subagents.episode_deadline_seconds == 2400.0
    assert [member.value for member in subagents.eligible_classes] == (
        AUTHORITATIVE_ALLOWLIST
    )


def test_schema_recovery_container_rejects_unknown_subsections_and_keys() -> None:
    # The [recovery] container is closed: only known subsections validate
    # (R2 adds [recovery.root] deliberately, never by accident).
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ChartreuxConfigSchema.model_validate({"recovery": {"bogus": {}}})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RecoverySectionConfig.model_validate({"bogus": {}})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ChartreuxConfigSchema.model_validate({
            "recovery": {"subagents": {"nonexistent_key": 1}}
        })
    # An empty container keeps the typed defaults (enabled since v0.5.6).
    empty = ChartreuxConfigSchema.model_validate({"recovery": {}})
    assert empty.recovery.subagents == RecoveryConfig()
    assert empty.recovery.subagents.enabled is True


def test_schema_rejects_invalid_nested_recovery_values() -> None:
    with pytest.raises(ValidationError, match="auto_retries_max"):
        ChartreuxConfigSchema.model_validate({
            "recovery": {"subagents": {"auto_retries_max": -1}}
        })
    with pytest.raises(ValidationError, match="eligible_classes"):
        ChartreuxConfigSchema.model_validate({
            "recovery": {"subagents": {"eligible_classes": ["bogus"]}}
        })
    with pytest.raises(ValidationError, match="nondecreasing"):
        ChartreuxConfigSchema.model_validate({
            "recovery": {"subagents": {"cooldowns_seconds": [120.0, 30.0]}}
        })


@pytest.mark.asyncio
async def test_layered_toml_deep_merge_preserves_unspecified_recovery_keys(
    tmp_path: Path,
) -> None:
    lower = tmp_path / "lower.toml"
    lower.write_text(
        """
[recovery.subagents]
cooldowns_seconds = [10.0, 20.0, 30.0]
jitter_fraction = 0.5
eligible_classes = ["timeout", "rate_limit"]
"""
    )
    higher = tmp_path / "higher.toml"
    higher.write_text(
        """
[recovery.subagents]
enabled = true
auto_retries_max = 1
"""
    )

    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([
        UserConfigLayer(path=lower, name="lower"),
        UserConfigLayer(path=higher, name="higher"),
    ])

    config = await builder.build()

    subagents = config.recovery.subagents
    assert subagents.enabled is True
    assert subagents.auto_retries_max == 1
    assert subagents.cooldowns_seconds == [10.0, 20.0, 30.0]
    assert subagents.jitter_fraction == 0.5
    assert subagents.eligible_classes == [
        RecoveryEligibleClass.TIMEOUT,
        RecoveryEligibleClass.RATE_LIMIT,
    ]
    # Untouched keys keep the typed defaults.
    assert subagents.episode_deadline_seconds == 2400.0
    assert subagents.recovery_concurrency == 2


def test_availability_registry_wiring_uses_the_subagents_correlation_thresholds() -> (
    None
):
    """WP4 (WP1 E4 #4): the shared registry is built from [recovery.subagents].

    The orchestrator's default AvailabilityRegistry carries the configured
    endpoint correlation thresholds (window, min events, min models, endpoint
    cooldown) instead of hard defaults; a non-Chartreux config keeps the
    plain defaults.
    """
    config = ChartreuxConfigSchema.model_validate({
        "recovery": {
            "subagents": {
                "endpoint_correlation_window_seconds": 111.0,
                "endpoint_correlation_min_events": 3,
                "endpoint_correlation_min_models": 4,
                "endpoint_cooldown_seconds": 7.5,
            }
        }
    })
    registry = _availability_registry_from_config(config)
    assert registry.endpoint_correlation_window == 111.0
    assert registry.endpoint_correlation_min_events == 3
    assert registry.endpoint_correlation_min_models == 4
    assert registry.endpoint_cooldown == 7.5

    defaults = _availability_registry_from_config(object())
    assert defaults.endpoint_correlation_window == 300.0
    assert defaults.endpoint_correlation_min_events == 2
    assert defaults.endpoint_correlation_min_models == 2
    assert defaults.endpoint_cooldown == 30.0
    assert isinstance(defaults, AvailabilityRegistry)


@pytest.mark.parametrize("count", [-1, 11])
def test_rejects_negative_and_over_limit_retry_counts(count: int) -> None:
    with pytest.raises(ValidationError, match="auto_retries_max"):
        RecoveryConfig(auto_retries_max=count)


@pytest.mark.parametrize("count", [True, 2.0, "2"])
def test_rejects_non_strict_integer_retry_counts(count: object) -> None:
    with pytest.raises(ValidationError, match="auto_retries_max"):
        RecoveryConfig.model_validate({"auto_retries_max": count})


@pytest.mark.parametrize(
    "deadline",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("-inf"), id="neg-inf"),
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
    ],
)
def test_rejects_non_finite_or_nonpositive_episode_deadline(deadline: float) -> None:
    with pytest.raises(ValidationError, match="episode_deadline_seconds"):
        RecoveryConfig(episode_deadline_seconds=deadline)


def test_rejects_unknown_eligible_class() -> None:
    with pytest.raises(ValidationError, match="eligible_classes"):
        RecoveryConfig.model_validate({"eligible_classes": ["not_a_class"]})


def test_rejects_enabled_controller_without_automatic_retries() -> None:
    with pytest.raises(ValidationError, match="requires auto_retries_max"):
        RecoveryConfig(enabled=True, auto_retries_max=0)


def test_rejects_decreasing_cooldowns() -> None:
    with pytest.raises(ValidationError, match="nondecreasing"):
        RecoveryConfig(cooldowns_seconds=[120.0, 30.0])


def test_rejects_fewer_cooldowns_than_automatic_retries() -> None:
    with pytest.raises(ValidationError, match="at least auto_retries_max"):
        RecoveryConfig(auto_retries_max=3)


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("-inf"), id="neg-inf"),
        pytest.param(-1.0, id="negative"),
    ],
)
def test_rejects_non_finite_or_negative_cooldown_entries(entry: float) -> None:
    with pytest.raises(ValidationError, match="must be finite, nonnegative"):
        RecoveryConfig(cooldowns_seconds=[30.0, entry])


@pytest.mark.parametrize(
    "jitter",
    [
        pytest.param(-0.1, id="negative"),
        pytest.param(1.5, id="above-one"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
    ],
)
def test_rejects_jitter_outside_the_unit_interval(jitter: float) -> None:
    with pytest.raises(ValidationError, match="jitter_fraction"):
        RecoveryConfig(jitter_fraction=jitter)


@pytest.mark.parametrize("concurrency", [0, -1])
def test_rejects_nonpositive_recovery_concurrency(concurrency: int) -> None:
    with pytest.raises(ValidationError, match="recovery_concurrency"):
        RecoveryConfig(recovery_concurrency=concurrency)


def test_rejects_negative_suspension_limits() -> None:
    with pytest.raises(ValidationError, match="suspension_ttl_seconds"):
        RecoveryConfig(suspension_ttl_seconds=-5.0)
    with pytest.raises(ValidationError, match="max_suspended_agents"):
        RecoveryConfig(max_suspended_agents=-1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("endpoint_correlation_min_events", 1),
        ("endpoint_correlation_min_models", 1),
        ("endpoint_correlation_window_seconds", 0.0),
        ("endpoint_correlation_window_seconds", float("nan")),
        ("endpoint_cooldown_seconds", -1.0),
    ],
)
def test_rejects_invalid_correlation_thresholds(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        RecoveryConfig.model_validate({field: value})


@pytest.mark.parametrize(
    "price",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("nan"), id="nan"),
    ],
)
def test_rejects_invalid_fallback_price_cap(price: float) -> None:
    with pytest.raises(ValidationError, match="fallback_max_output_price_per_million"):
        RecoveryConfig(fallback_max_output_price_per_million=price)


def test_rejects_unknown_config_keys() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RecoveryConfig.model_validate({"nonexistent_key": 1})


def test_accepts_zero_retries_and_empty_cooldowns_when_disabled() -> None:
    config = RecoveryConfig(enabled=False, auto_retries_max=0, cooldowns_seconds=[])

    assert config.auto_retries_max == 0
    assert config.cooldowns_seconds == []


def test_accepts_cooldowns_matching_the_retry_count() -> None:
    config = RecoveryConfig(auto_retries_max=3, cooldowns_seconds=[30.0, 60.0, 120.0])

    assert config.cooldowns_seconds == [30.0, 60.0, 120.0]


def test_accepts_zero_suspension_limits() -> None:
    config = RecoveryConfig(suspension_ttl_seconds=0.0, max_suspended_agents=0)

    assert config.suspension_ttl_seconds == 0.0
    assert config.max_suspended_agents == 0


def test_accepts_eligible_fallback_constraints() -> None:
    config = RecoveryConfig(
        fallback_require_endpoint_independence=False,
        fallback_allowed_base_models=["glm-5-3"],
        fallback_max_output_price_per_million=2.5,
    )

    assert config.fallback_require_endpoint_independence is False
    assert config.fallback_allowed_base_models == ["glm-5-3"]
    assert config.fallback_max_output_price_per_million == 2.5


def test_recovery_config_is_exported_from_the_config_package() -> None:
    import chartreux.core.config as config_package
    from chartreux.core.config import (
        RecoveryConfig as ExportedConfig,
        RecoveryEligibleClass as ExportedClass,
        RecoverySectionConfig as ExportedSection,
    )

    assert ExportedConfig is RecoveryConfig
    assert ExportedClass is RecoveryEligibleClass
    assert ExportedSection is RecoverySectionConfig
    assert "RecoveryConfig" in config_package.__all__
    assert "RecoveryEligibleClass" in config_package.__all__
    assert "RecoverySectionConfig" in config_package.__all__
