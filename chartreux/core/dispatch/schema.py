"""Immutable, TOML-shaped dispatch policy definitions.

Reference integrity and activation eligibility belong to dispatch lint, not
these structural models. Catalog loading and revision integration are separate.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

PurposeId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$")]
DispatchName = Annotated[str, Field(pattern=r"^[a-zA-Z0-9_.-]+$")]
ImplementationClass = Literal["never", "routine", "escalation"]


class DispatchMode(StrEnum):
    STANDALONE = "standalone"
    ORCHESTRATED = "orchestrated"


class _FrozenDispatchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> _FrozenDispatchModel:
        return self


class PurposeDefinition(_FrozenDispatchModel):
    """Developer-owned rendering for one purpose, including overlay additions."""

    description: str = Field(min_length=1)

    @field_validator("description")
    @classmethod
    def description_is_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("purpose description must not be blank")
        return value


PurposeVocabulary = Mapping[PurposeId, PurposeDefinition]


def _shipped_vocabulary() -> PurposeVocabulary:
    from chartreux.core.dispatch.purposes import SHIPPED_PURPOSES

    return SHIPPED_PURPOSES


class DispatchSlot(_FrozenDispatchModel):
    """A named launch binding; neither a role nor a canonical model identity."""

    profile: DispatchName
    role: str = Field(pattern=r"^@[a-zA-Z0-9_.-]+$")
    purposes: tuple[PurposeId, ...]
    implements: ImplementationClass
    review_eligible: bool

    @field_validator("profile")
    @classmethod
    def profile_is_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("profile must not be blank")
        return value

    @field_validator("purposes")
    @classmethod
    def purposes_are_known(
        cls, value: tuple[PurposeId, ...], info: ValidationInfo
    ) -> tuple[PurposeId, ...]:
        known = (info.context or {}).get("purpose_ids", _shipped_vocabulary())
        if unknown := set(value) - set(known):
            raise ValueError(f"unknown dispatch purposes: {sorted(unknown)!r}")
        return value


class DispatchPolicy(_FrozenDispatchModel):
    """Shipped mode selection, named slots, vocabulary, and curated prose.

    Dumped dictionaries contain only ordinary TOML/JSON-compatible containers.
    Hash the canonical JSON dump alongside catalog data when wiring revisions.
    """

    mode: DispatchMode
    identity: str = Field(min_length=1)
    version: int = Field(strict=True, ge=1)
    vocabulary: PurposeVocabulary = Field(default_factory=_shipped_vocabulary)
    slots: Mapping[DispatchName, DispatchSlot]
    instructions: str = ""
    failure_routing: str = ""
    compositions: str = ""
    contrasts: str = ""

    @model_validator(mode="before")
    @classmethod
    def validate_slots_with_vocabulary(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        vocabulary = value.get("vocabulary", _shipped_vocabulary())
        slots = value.get("slots")
        if isinstance(vocabulary, Mapping) and isinstance(slots, Mapping):
            return {
                **value,
                "slots": {
                    name: DispatchSlot.model_validate(
                        slot, context={"purpose_ids": vocabulary}
                    )
                    for name, slot in slots.items()
                },
            }
        return value

    @field_validator("identity")
    @classmethod
    def identity_is_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("policy identity must not be blank")
        return value

    @field_validator("vocabulary", "slots")
    @classmethod
    def freeze_tables(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return MappingProxyType(dict(value))

    @model_validator(mode="after")
    def slot_purposes_exist(self) -> DispatchPolicy:
        for name, slot in self.slots.items():
            if not name.strip():
                raise ValueError("slot name must not be blank")
            if unknown := set(slot.purposes) - set(self.vocabulary):
                raise ValueError(
                    f"slot {name!r} references unknown purposes: {sorted(unknown)!r}"
                )
        return self

    @field_serializer("mode")
    def serialize_mode(self, value: DispatchMode) -> str:
        return value.value

    @field_serializer("vocabulary", "slots")
    def serialize_tables(self, value: Mapping[str, BaseModel]) -> dict[str, Any]:
        return {name: entry.model_dump(mode="json") for name, entry in value.items()}
