from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import tomllib
from typing import Any

from pydantic import ValidationError
import pytest
import tomli_w

from chartreux.core.dispatch import (
    DEFAULT_DISPATCH_MODE,
    ORCHESTRATED_PRESET,
    SHIPPED_PRESETS,
    SHIPPED_PURPOSES,
    STANDALONE_PRESET,
    DispatchMode,
    DispatchPolicy,
    DispatchSlot,
    PurposeDefinition,
)
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG


def _slot_data() -> dict[str, Any]:
    return {
        "profile": "worker",
        "role": "@medium",
        "purposes": ["implementation"],
        "implements": "routine",
        "review_eligible": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unknown", "value"),
        ("purposes", ["unknown-purpose"]),
        ("implements", "sometimes"),
        ("role", "medium"),
        ("role", "@"),
        ("profile", " "),
    ],
)
def test_slot_rejects_invalid_data(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        DispatchSlot.model_validate({**_slot_data(), field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unknown", "value"),
        ("mode", "implement directly and delegate as needed"),
        ("mode", "custom"),
        ("version", 0),
        ("version", True),
        ("identity", " "),
    ],
)
def test_policy_rejects_invalid_data(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        DispatchPolicy.model_validate({
            **ORCHESTRATED_PRESET.model_dump(),
            field: value,
        })


def test_policy_rejects_unknown_nested_purpose_and_fields() -> None:
    data = ORCHESTRATED_PRESET.model_dump()
    data["slots"]["implementor"]["purposes"] = ["typo"]
    with pytest.raises(ValidationError, match="unknown dispatch purposes"):
        DispatchPolicy.model_validate(data)
    data["slots"]["implementor"] = {**_slot_data(), "description": "per-slot prose"}
    with pytest.raises(ValidationError):
        DispatchPolicy.model_validate(data)
    with pytest.raises(ValidationError):
        PurposeDefinition.model_validate({"description": "Valid", "unknown": True})


def test_slot_toml_round_trip_and_immutability() -> None:
    data = _slot_data()
    slot = DispatchSlot.model_validate(data)
    data["purposes"].append("search")
    assert slot.purposes == ("implementation",)
    restored = DispatchSlot.model_validate(
        tomllib.loads(tomli_w.dumps(slot.model_dump()))
    )
    assert restored == slot
    assert json.loads(slot.model_dump_json())["purposes"] == ["implementation"]
    with pytest.raises(ValidationError, match="frozen"):
        slot.role = "@small"
    assert deepcopy(slot) is slot


@pytest.mark.parametrize("policy", list(SHIPPED_PRESETS.values()))
def test_shipped_policy_round_trip_and_immutability(policy: DispatchPolicy) -> None:
    dumped = policy.model_dump()
    assert DispatchPolicy.model_validate(tomllib.loads(tomli_w.dumps(dumped))) == policy
    assert DispatchPolicy.model_validate_json(policy.model_dump_json()) == policy
    assert dumped == json.loads(policy.model_dump_json())
    with pytest.raises(TypeError):
        policy.slots["new"] = policy.slots["implementor"]  # type: ignore[index]
    with pytest.raises(TypeError):
        policy.vocabulary["new"] = PurposeDefinition(description="New")  # type: ignore[index]
    with pytest.raises(ValidationError, match="frozen"):
        policy.version = 2
    assert deepcopy(policy) is policy
    for slot in policy.slots.values():
        assert slot.role[1:] in SHIPPED_CATALOG.roles
        assert set(slot.purposes) <= set(policy.vocabulary)
    assert {
        purpose for slot in policy.slots.values() for purpose in slot.purposes
    } == set(SHIPPED_PURPOSES)


def test_overlay_purpose_requires_an_entry_with_rendering() -> None:
    data = ORCHESTRATED_PRESET.model_dump()
    data["slots"]["custom"] = {**_slot_data(), "purposes": ["custom-work"]}
    with pytest.raises(ValidationError):
        DispatchPolicy.model_validate(data)
    data["vocabulary"]["custom-work"] = {"description": "Perform bounded custom work."}
    policy = DispatchPolicy.model_validate(data)
    assert policy.slots["custom"].purposes == ("custom-work",)
    assert (
        DispatchPolicy.model_validate(tomllib.loads(tomli_w.dumps(policy.model_dump())))
        == policy
    )
    data["vocabulary"]["custom-work"] = {"description": " "}
    with pytest.raises(ValidationError):
        DispatchPolicy.model_validate(data)
    data["vocabulary"]["custom-work"] = {}
    with pytest.raises(ValidationError):
        DispatchPolicy.model_validate(data)


def test_vocabulary_has_all_developer_owned_descriptions() -> None:
    assert set(SHIPPED_PURPOSES) == {
        "search",
        "exploration",
        "verification",
        "mechanical-edit",
        "implementation",
        "implementation-demanding-settled",
        "design-analysis",
        "planning-analysis",
        "review.quick",
        "review.standard",
        "review.deep",
    }
    assert all(entry.description.strip() for entry in SHIPPED_PURPOSES.values())
    assert "never implementation" in SHIPPED_PURPOSES["design-analysis"].description
    assert "never implementation" in SHIPPED_PURPOSES["planning-analysis"].description
    assert (
        "settled approach"
        in SHIPPED_PURPOSES["implementation-demanding-settled"].description
    )


def test_presets_and_legacy_curated_blocks() -> None:
    assert DEFAULT_DISPATCH_MODE == DispatchMode.STANDALONE
    assert set(SHIPPED_PRESETS) == {"standalone", "orchestrated"}
    # The WP0 byte-equality baseline (ADR 0018-G.1) is superseded: the roster
    # rename rewrote the curated prose blocks, and the regenerated active
    # goldens are the byte baseline now. The legacy captures stay untouched.
    assert (
        "The shipped model roles are `orchestrator` for the main assistant and "
        "`worker`, `scout`, and `heavy` for subagent work."
        in ORCHESTRATED_PRESET.instructions
    )
    assert 'config={"model": "@scout"}' in ORCHESTRATED_PRESET.contrasts
    assert "retry at `@worker` with a different approach" in (
        ORCHESTRATED_PRESET.failure_routing
    )
    assert "You may implement directly:" in STANDALONE_PRESET.instructions
    assert (
        "Never run tests, builds, or other verification yourself"
        in STANDALONE_PRESET.instructions
    )
    assert (
        "The opening gate and acceptance lattice remain unchanged"
        in STANDALONE_PRESET.instructions
    )
    assert "apply next session" in STANDALONE_PRESET.instructions


def test_policy_is_revision_hashable_as_canonical_data() -> None:
    def revision(data: dict[str, Any]) -> str:
        return sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    data = ORCHESTRATED_PRESET.model_dump()
    assert revision(data) == revision(json.loads(ORCHESTRATED_PRESET.model_dump_json()))
    changed = deepcopy(data)
    changed["slots"]["implementor"]["role"] = "@small"
    assert revision(changed) != revision(data)
    changed = deepcopy(data)
    changed["contrasts"] += " Changed."
    assert revision(changed) != revision(data)


def test_shipped_model_payload_and_role_bindings_preserved() -> None:
    # GLM payload captured from WP0 HEAD (30751cb), before thinning
    # descriptions.
    assert SHIPPED_CATALOG.models["glm-5-3"].model_dump() == {
        "thinking": "high",
        "temperature": None,
        "disabled": False,
        "deployments": (
            {
                "provider": "mistral",
                "name": "zai-glm-5-3",
                "prices": {"input": 1.4, "output": 4.4, "cached_input": 0.14},
                "supports_images": False,
                "supported_thinking_levels": None,
                "auto_compact_threshold": 400000,
                "disabled": False,
            },
        ),
    }
    # The ML4 entry ships the runtime-confirmed wire ID at launch-sale prices.
    assert SHIPPED_CATALOG.models["mistral-large-4"].model_dump() == {
        "thinking": "high",
        "temperature": None,
        "disabled": False,
        "deployments": (
            {
                "provider": "mistral",
                "name": "mistral-large-4",
                "prices": {"input": 0.68, "output": 2.09, "cached_input": 0.07},
                "supports_images": True,
                "supported_thinking_levels": None,
                "auto_compact_threshold": 400000,
                "disabled": False,
            },
        ),
    }
    assert set(SHIPPED_CATALOG.models) == {"glm-5-3", "mistral-large-4"}
    assert {
        name: (role.model, role.thinking)
        for name, role in SHIPPED_CATALOG.roles.items()
    } == {
        "orchestrator": ("glm-5-3", "high"),
        "worker": ("glm-5-3", "medium"),
        "scout": ("glm-5-3", "low"),
        "heavy": ("mistral-large-4", "high"),
    }
    assert {role.description for role in SHIPPED_CATALOG.roles.values()} == {
        "main assistant preset",
        "worker preset",
        "scout preset",
        "heavy preset",
    }
    slots = ORCHESTRATED_PRESET.slots
    assert "implementor" != slots["implementor"].role[1:]
    assert slots["implementor"].profile == "worker"
    bindings = {
        (
            SHIPPED_CATALOG.roles[slot.role[1:]].model,
            SHIPPED_CATALOG.roles[slot.role[1:]].thinking,
        )
        for slot in slots.values()
    }
    assert len({model for model, _ in bindings}) == 2
    assert len(bindings) == 3
