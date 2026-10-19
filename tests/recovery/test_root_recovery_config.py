"""R2 — the validated ``[recovery.root]`` config section (plan state.md R2).

The root-turn automatic recovery controller's policy lives in the
``[recovery.root]`` subsection of the ``[recovery]`` container, symmetric with
``[recovery.subagents]`` (WP4) and ``[user_questions]`` (U). It declares the
ONE shared automatic-turn budget (merging the legacy incomplete-stream
continuation), the advisory episode deadline, the cooldown-aware backoff
schedule, jitter, the read-only availability re-check floor, and the eligible
classes (the ONE authoritative WP1b transient allowlist, restricted by
type).

AUTOMATIC RECOVERY IS ENABLED BY DEFAULT since v0.5.6: ``enabled`` defaults to
True (promoted at WP6 per OD4b once R2's acceptance matrix passed). OD4a
REMOVAL-BY-DEFAULT: when ``enabled`` is False, NO root automatic continuation
runs at all — including the migrated incomplete-stream path (a documented
breaking change).

Round-2 amendment: the disabled-state tests set ``enabled`` EXPLICITLY (never
relying on the shipped default — WP6 promotes it and must not break them);
the WP6 enablement matrix gains an explicit enabled-default behavior case for
``[recovery.root]``. Strict validation rejects negative counts, NaN/infinite
durations, unknown fields/classes, and contradictory constraints. The
aggregate ``deadline_seconds`` is an ADVISORY scheduling/admission bound
(Q2 negative: never an in-flight kill).
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import ValidationError
import pytest

from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage, UserCommandMessage
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import (
    RecoveryEligibleClass,
    RecoverySectionConfig,
    RootRecoveryConfig,
)
from tests.conftest import (
    build_test_agent_loop,
    build_test_chartreux_app,
    build_test_vibe_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


async def _wait_until(pilot: Any, predicate: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for UI state")
        await pilot.pause(0.01)


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


def test_root_automatic_recovery_is_enabled_by_default() -> None:
    # WP6 default promotion (OD4b): the shipped default is enabled.
    assert RootRecoveryConfig().enabled is True
    assert ChartreuxConfigSchema().recovery.root.enabled is True
    # The container's default is a fully default (enabled) root policy.
    assert ChartreuxConfigSchema().recovery.root == RootRecoveryConfig()


def test_root_default_policy_matches_the_plan_values() -> None:
    config = RootRecoveryConfig()

    assert config.auto_retries_max == 2
    assert config.deadline_seconds == 2400.0
    assert config.cooldowns_seconds == [30.0, 120.0]
    assert config.jitter_fraction == 0.2
    assert config.availability_recheck_seconds == 1.0
    assert [member.value for member in config.eligible_classes] == (
        AUTHORITATIVE_ALLOWLIST
    )


def test_root_section_sits_beside_subagents_with_deep_merge() -> None:
    config = ChartreuxConfigSchema.model_validate({
        "recovery": {"root": {"enabled": True}}
    })

    root = config.recovery.root
    assert root.enabled is True
    # A sparse layer setting only [recovery.root] enabled preserves every
    # other typed default (deep-merge semantics, symmetric with subagents).
    assert root.auto_retries_max == 2
    assert root.deadline_seconds == 2400.0
    assert root.cooldowns_seconds == [30.0, 120.0]
    assert root.jitter_fraction == 0.2
    assert root.availability_recheck_seconds == 1.0
    assert [member.value for member in root.eligible_classes] == (
        AUTHORITATIVE_ALLOWLIST
    )
    # The sibling subsection is untouched by the root layer (enabled since
    # the v0.5.6 default promotion, OD5).
    assert config.recovery.subagents.enabled is True


def test_root_container_rejects_unknown_subsections_and_fields() -> None:
    with pytest.raises(ValidationError):
        RecoverySectionConfig.model_validate({"unknown": {}})
    with pytest.raises(ValidationError):
        RootRecoveryConfig.model_validate({"unknown_field": 1})
    with pytest.raises(ValidationError):
        ChartreuxConfigSchema.model_validate({
            "recovery": {"root": {"unknown_field": 1}}
        })


def test_root_eligible_classes_are_restricted_to_the_transient_allowlist() -> None:
    # The type restriction IS the safety exclusion: protected and unknown
    # classes can never be configured as recovery-eligible.
    for excluded in ("conflict", "too_early", "unknown", "cancellation", "auth"):
        with pytest.raises(ValidationError, match="eligible_classes"):
            RootRecoveryConfig.model_validate({"eligible_classes": [excluded]})

    coerced = RootRecoveryConfig.model_validate({
        "eligible_classes": ["timeout", "empty_response"]
    })
    assert coerced.eligible_classes == [
        RecoveryEligibleClass.TIMEOUT,
        RecoveryEligibleClass.EMPTY_RESPONSE,
    ]


def test_root_enabled_requires_a_usable_budget() -> None:
    with pytest.raises(ValidationError, match="auto_retries_max"):
        RootRecoveryConfig.model_validate({"enabled": True, "auto_retries_max": 0})
    with pytest.raises(ValidationError, match="eligible class"):
        RootRecoveryConfig.model_validate({"enabled": True, "eligible_classes": []})
    # Disabled with a zero budget is coherent (the controller never runs).
    assert (
        RootRecoveryConfig.model_validate({
            "enabled": False,
            "auto_retries_max": 0,
        }).auto_retries_max
        == 0
    )


@pytest.mark.parametrize("field", ["auto_retries_max"])
def test_root_count_bounds(field: str) -> None:
    with pytest.raises(ValidationError):
        RootRecoveryConfig.model_validate({field: -1})
    with pytest.raises(ValidationError):
        RootRecoveryConfig.model_validate({field: 11})
    with pytest.raises(ValidationError):
        RootRecoveryConfig.model_validate({field: 2.5})


@pytest.mark.parametrize(
    "field,value",
    [
        ("deadline_seconds", 0.0),
        ("deadline_seconds", 86400.1),
        ("deadline_seconds", float("nan")),
        ("deadline_seconds", float("inf")),
        ("deadline_seconds", True),
        ("jitter_fraction", -0.1),
        ("jitter_fraction", 1.1),
        ("jitter_fraction", float("nan")),
        ("availability_recheck_seconds", 0.0),
        ("availability_recheck_seconds", -1.0),
        ("availability_recheck_seconds", float("inf")),
    ],
)
def test_root_duration_and_jitter_validation(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        RootRecoveryConfig.model_validate({field: value})


def test_root_cooldowns_must_be_finite_nonnegative_nondecreasing_and_enough() -> None:
    with pytest.raises(ValidationError, match="cooldowns_seconds"):
        RootRecoveryConfig.model_validate({"cooldowns_seconds": [-1.0, 30.0]})
    with pytest.raises(ValidationError, match="cooldowns_seconds"):
        RootRecoveryConfig.model_validate({"cooldowns_seconds": [float("nan"), 30.0]})
    with pytest.raises(ValidationError, match="cooldowns_seconds"):
        RootRecoveryConfig.model_validate({"cooldowns_seconds": [float("inf"), 30.0]})
    with pytest.raises(ValidationError, match="nondecreasing"):
        RootRecoveryConfig.model_validate({"cooldowns_seconds": [120.0, 30.0]})
    with pytest.raises(ValidationError, match="at least auto_retries_max"):
        RootRecoveryConfig.model_validate({
            "auto_retries_max": 3,
            "cooldowns_seconds": [30.0, 120.0],
        })
    # Enough entries for the retry count is accepted.
    assert RootRecoveryConfig.model_validate({
        "auto_retries_max": 3,
        "cooldowns_seconds": [30.0, 120.0, 300.0],
    }).cooldowns_seconds == [30.0, 120.0, 300.0]


def test_root_recheck_floor_must_fit_the_deadline() -> None:
    with pytest.raises(ValidationError, match="availability_recheck_seconds"):
        RootRecoveryConfig.model_validate({
            "deadline_seconds": 1.0,
            "availability_recheck_seconds": 2.0,
        })
    assert (
        RootRecoveryConfig.model_validate({
            "deadline_seconds": 10.0,
            "availability_recheck_seconds": 2.0,
        }).availability_recheck_seconds
        == 2.0
    )


def test_root_enabled_is_a_strict_bool() -> None:
    with pytest.raises(ValidationError):
        RootRecoveryConfig.model_validate({"enabled": "yes"})
    with pytest.raises(ValidationError):
        RootRecoveryConfig.model_validate({"enabled": 1})


@pytest.mark.asyncio
async def test_disabled_root_recovery_runs_no_automatic_continuation() -> None:
    """Item 20 (behavioral half) + the round-2 amendment: with
    ``[recovery.root] enabled=False`` set EXPLICITLY (never relying on the
    development default), a terminal incomplete-stream failure runs NO
    automatic continuation at all — not even the migrated legacy path (OD4a
    removal-by-default) — and the manual R1 retry affordance is offered
    instead.
    """
    backend = FakeBackend([
        [mock_llm_chunk(content="partial", stop_reason=None)],
        [mock_llm_chunk(content="never used")],
    ])
    config = build_test_vibe_config().model_copy(
        update={
            "recovery": RecoverySectionConfig(root=RootRecoveryConfig(enabled=False))
        }
    )
    agent_loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=True
    )
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        # The terminal failure surfaces with the manual R1 retry affordance.
        await _wait_until(pilot, lambda: len(app.query(ErrorMessage)) == 1)
        await _wait_until(pilot, lambda: not app._agent_job_active())
        # The legacy path continued IMMEDIATELY; a short settle window proves
        # no hidden automatic continuation is armed anywhere.
        await pilot.pause(0.2)

        # Exactly one request: no hidden legacy auto-continuation (OD4a).
        assert len(backend.requests_messages) == 1
        assert len(app.query(ErrorMessage)) == 1
        assert "/retry" in str(app.query_one(ErrorMessage)._error)
        # No recovery milestone notices were emitted.
        notices = " ".join(
            str(message._content) for message in app.query(UserCommandMessage)
        )
        assert "Automatic retry" not in notices
