"""Package 8 — the canonical task grammar (chartreux.core.agents.identity_launch).

Tests the grammar MODULE — the canonical schema, the shorthand parser, and the
cross-field validation — NOT ``TaskArgs`` (the ``TaskArgs`` rewiring onto this
schema belongs to package 11b; this module is written to be consumable by it).
"""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from chartreux.core.agents.identity_launch import (
    ModelShorthand,
    TaskGrammarError,
    TaskIdentityArgs,
    parse_model_shorthand,
    validate_identity_reference,
)

# ---------------------------------------------------------------------------
# The canonical top-level schema
# ---------------------------------------------------------------------------


def test_canonical_top_level_schema_fields() -> None:
    args = TaskIdentityArgs.model_validate({
        "task": "research the thing",
        "model": "@fast",
        "identity": "researcher",
        "purpose": "research",
    })
    assert (args.model, args.identity, args.purpose) == (
        "@fast",
        "researcher",
        "research",
    )
    # The lifecycle defaults are unchanged.
    assert args.background is True
    assert args.replace_run is False
    assert args.retain is None
    assert args.agent_id is None
    assert args.task_summary is None
    # The schema round-trips.
    assert TaskIdentityArgs.model_validate(args.model_dump()) == args


def test_unknown_fields_are_rejected() -> None:
    # The legacy agent_type bridge is NOT part of the canonical schema (it
    # is retired with the profile axis, package 16a).
    with pytest.raises(ValidationError):
        TaskIdentityArgs.model_validate({"task": "x", "agent_type": "worker"})


# ---------------------------------------------------------------------------
# The shorthand grammar: @<preset-or-canonical-model>[:<identity>]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("@fast:researcher", ModelShorthand("@fast", "researcher")),
        ("@fast:user:foo", ModelShorthand("@fast", "user:foo")),
        ("@fast", ModelShorthand("@fast", None)),
        ("@mistral-large-4:reviewer", ModelShorthand("@mistral-large-4", "reviewer")),
    ],
)
def test_parse_shorthand(value: str, expected: ModelShorthand) -> None:
    assert parse_model_shorthand(value) == expected


@pytest.mark.parametrize("value", ["fast", "mistral-large-4"])
def test_plain_canonical_model_expressions_are_not_shorthand(value: str) -> None:
    assert parse_model_shorthand(value) is None


@pytest.mark.parametrize("value", ["@orchestrator", "@main"])
def test_role_expressions_parse_with_the_identity_component_absent(value: str) -> None:
    # Missing components are never guessed: the identity component is absent.
    assert parse_model_shorthand(value) == ModelShorthand(value, None)


@pytest.mark.parametrize(
    "value",
    [
        "@",  # empty everything
        "@:",  # empty model + empty identity
        "@:researcher",  # empty model component
        "@fast:",  # empty identity component
        "@@fast",  # nested '@' model component
        "@fast:builtin:coder",  # builtin: is structured-field-only
        "@fast:user:foo:bar",  # ambiguous multi-colon
        "@fast:researcher:extra",  # ambiguous multi-colon
        "@fast:user:",  # empty custom name
        "@fast:user:../evil",  # traversal
        "@fast:user:a/b",  # path separator
        "@fast:notabuiltin",  # a bare custom id is not a shorthand component
        "fast:researcher",  # a colon without '@' is ambiguous
    ],
)
def test_malformed_or_ambiguous_shorthand_is_rejected(value: str) -> None:
    with pytest.raises(TaskGrammarError) as excinfo:
        parse_model_shorthand(value)
    assert excinfo.value.field == "model"
    assert value in str(excinfo.value)


@pytest.mark.parametrize(
    "model",
    [
        "@fast:",
        "@:researcher",
        "@fast:user:a:b",
        "@fast:builtin:coder",
        "fast:researcher",
        "@fast:notabuiltin",
    ],
)
def test_malformed_model_field_is_rejected_by_the_schema(model: str) -> None:
    with pytest.raises(ValidationError):
        TaskIdentityArgs.model_validate({"task": "x", "model": model})


# ---------------------------------------------------------------------------
# Cross-field grammar: conflicts and dual supply
# ---------------------------------------------------------------------------


def test_dual_top_level_and_nested_model_supply_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        TaskIdentityArgs.model_validate({
            "task": "x",
            "model": "@fast",
            "config": {"model": "fast"},
        })
    assert "supply exactly one" in str(excinfo.value)


def test_shorthand_and_structured_identity_conflict_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        TaskIdentityArgs.model_validate({
            "task": "x",
            "model": "@fast:researcher",
            "identity": "coder",
        })
    assert "carries an identity component" in str(excinfo.value)


def test_shorthand_model_with_structured_identity_composes() -> None:
    args = TaskIdentityArgs.model_validate({
        "task": "x",
        "model": "@fast",
        "identity": "researcher",
    })
    assert (args.model, args.identity) == ("@fast", "researcher")


def test_seat_without_purpose_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskIdentityArgs.model_validate({"task": "x", "seat": "analytical"})


# ---------------------------------------------------------------------------
# The identity component: bare built-ins and user:-qualified ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "identity",
    [
        "coder",
        "researcher",
        "reviewer",
        "advisor",
        "orchestrator",
        "builtin:coder",
        "builtin:orchestrator",
        "user:field-notes",
        "field-notes",
    ],
)
def test_valid_identity_reference_forms(identity: str) -> None:
    assert validate_identity_reference(identity) == identity
    args = TaskIdentityArgs.model_validate({"task": "x", "identity": identity})
    assert args.identity == identity


@pytest.mark.parametrize(
    "identity",
    [
        "",
        "user:",
        "user:.",
        "user:..",
        "user:a:b",
        "user:coder",  # reserved shipped name
        "user:reviewer",  # reserved shipped name
        "builtin:",
        "builtin:unknown",
        "builtin:a:b",
        "bad:id",
        "a/b",
        "a\\b",
    ],
)
def test_malformed_identity_references_are_rejected(identity: str) -> None:
    with pytest.raises(ValidationError):
        TaskIdentityArgs.model_validate({"task": "x", "identity": identity})
    with pytest.raises(TaskGrammarError):
        validate_identity_reference(identity)


# ---------------------------------------------------------------------------
# Lifecycle fields and the preserved nested config
# ---------------------------------------------------------------------------


def test_lifecycle_fields_and_additive_config_are_preserved() -> None:
    args = TaskIdentityArgs.model_validate({
        "task": "do work",
        "task_summary": "  a   summary  ",
        "background": False,
        "retain": True,
        "agent_id": "agent-7",
        "replace_run": True,
        "config": {
            "instructions": "additive",
            "work_package_id": "wp-1",
            "predecessor_agent_id": "agent-3",
            "package_tier": "mechanical",
        },
    })
    assert args.task_summary == "a summary"
    assert args.background is False
    assert args.retain is True
    assert args.agent_id == "agent-7"
    assert args.replace_run is True
    assert args.config is not None
    assert args.config.instructions == "additive"
    assert args.config.work_package_id == "wp-1"
    assert args.config.predecessor_agent_id == "agent-3"
    assert args.config.package_tier is not None
    assert args.config.package_tier.value == "mechanical"


def test_config_accepts_a_json_string_like_the_task_tool() -> None:
    args = TaskIdentityArgs.model_validate({
        "task": "x",
        "config": '{"instructions": "from json"}',
    })
    assert args.config is not None
    assert args.config.instructions == "from json"
    with pytest.raises(ValidationError):
        TaskIdentityArgs.model_validate({"task": "x", "config": "not json"})
    with pytest.raises(ValidationError):
        TaskIdentityArgs.model_validate({"task": "x", "config": "[1, 2]"})


# ---------------------------------------------------------------------------
# Consumability by the task tool (the TaskArgs surface)
# ---------------------------------------------------------------------------


def test_task_args_rides_on_the_canonical_schema() -> None:
    """Package 16a: TaskArgs IS the canonical schema — the legacy
    ``agent_type`` bridge field is retired with the profile axis.
    """
    from chartreux.core.subagents import TaskArgs

    args = TaskArgs.model_validate({
        "task": "x",
        "identity": "researcher",
        "purpose": "research",
    })
    assert (args.identity, args.purpose) == ("researcher", "research")
    assert isinstance(args, TaskIdentityArgs)
    assert "agent_type" not in TaskArgs.model_fields


def test_the_grammar_module_exposes_the_resolution_contract() -> None:
    """11b/11c consume the schema, the parser, the resolver, and the
    resolved-launch contract from the grammar module and launch_types.
    """
    from chartreux.core.agents import identity_launch
    from chartreux.core.launch_types import ResolvedIdentityLaunch

    assert identity_launch.TaskIdentityArgs is TaskIdentityArgs
    assert callable(identity_launch.resolve_identity_launch)
    assert identity_launch.ResolvedIdentityLaunch is ResolvedIdentityLaunch


# ---------------------------------------------------------------------------
# P1-c2: purpose-keyed difficulty admission
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "difficulty", ["novel-algorithm", "concurrency", "subtle-invariant"]
)
def test_demanding_route_accepts_a_qualifying_category(difficulty: str) -> None:
    from chartreux.core.launch_types import DifficultyCategory

    args = TaskIdentityArgs.model_validate({
        "task": "x",
        "purpose": "implementation-demanding-settled",
        "config": {"difficulty": difficulty},
    })
    assert args.config is not None
    assert isinstance(args.config.difficulty, DifficultyCategory)
    assert args.config.difficulty.value == difficulty


@pytest.mark.parametrize("difficulty", ["cross-subsystem", "none-of-these-fit"])
def test_demanding_route_record_only_category_fails_admission(difficulty: str) -> None:
    with pytest.raises(ValidationError, match="record-only"):
        TaskIdentityArgs.model_validate({
            "task": "x",
            "purpose": "implementation-demanding-settled",
            "config": {"difficulty": difficulty},
        })


def test_demanding_route_without_a_category_fails_admission() -> None:
    with pytest.raises(ValidationError, match="requires config.difficulty"):
        TaskIdentityArgs.model_validate({
            "task": "x",
            "purpose": "implementation-demanding-settled",
        })


@pytest.mark.parametrize("purpose", [None, "research", "implementation"])
def test_other_purposes_reject_the_difficulty_field(purpose: str | None) -> None:
    payload: dict[str, object] = {
        "task": "x",
        "config": {"difficulty": "novel-algorithm"},
    }
    if purpose is not None:
        payload["purpose"] = purpose
    with pytest.raises(ValidationError, match="only accepted for"):
        TaskIdentityArgs.model_validate(payload)


def test_task_args_inherits_the_difficulty_admission() -> None:
    """The task tool's surface is the canonical schema: TaskArgs enforces the
    same purpose-keyed difficulty admission on every launch path.
    """
    from chartreux.core.subagents import TaskArgs

    args = TaskArgs.model_validate({
        "task": "x",
        "purpose": "implementation-demanding-settled",
        "config": {"difficulty": "concurrency"},
    })
    assert args.config is not None
    assert args.config.difficulty is not None
    assert args.config.difficulty.value == "concurrency"
    with pytest.raises(ValidationError, match="record-only"):
        TaskArgs.model_validate({
            "task": "x",
            "purpose": "implementation-demanding-settled",
            "config": {"difficulty": "cross-subsystem"},
        })
