"""Purpose vocabulary and shipped catalog payload pins.

The dispatch slot/preset/schema surface is retired (package 16b): the
``DispatchMode``/``DispatchPolicy``/``DispatchSlot`` models and the shipped
presets are gone with the ``chartreux.core.dispatch.schema``/``presets``
modules. What survives on this axis is the purpose vocabulary (rendered into
the composed artifact's "Purpose meanings" table and keyed by the purpose
routes) and the shipped model catalog payload.
"""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from chartreux.core.dispatch import SHIPPED_PURPOSES, PurposeDefinition
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG


def test_purpose_definition_rejects_unknown_fields_and_blank_descriptions() -> None:
    with pytest.raises(ValidationError):
        PurposeDefinition.model_validate({"description": "Valid", "unknown": True})
    with pytest.raises(ValidationError):
        PurposeDefinition.model_validate({"description": " "})
    with pytest.raises(ValidationError):
        PurposeDefinition.model_validate({})
    definition = PurposeDefinition(description="Bounded custom work.")
    assert definition.description == "Bounded custom work."
    with pytest.raises(ValidationError, match="frozen"):
        definition.description = "Changed."  # type: ignore[misc]


def test_vocabulary_has_all_developer_owned_descriptions() -> None:
    assert set(SHIPPED_PURPOSES) == {
        "research",
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
        "main": ("glm-5-3", "high"),
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
