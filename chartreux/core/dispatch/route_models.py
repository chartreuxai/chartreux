from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PackageTier(StrEnum):
    MECHANICAL = "mechanical"
    SUBSTANTIVE = "substantive"


class ReplacementCapability(StrEnum):
    NONE = "none"
    MECHANICAL = "mechanical"
    SUBSTANTIVE = "substantive"


class RouteTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    identity: str
    tier: PackageTier
    seat: str | None = None
    overlay: str | None = None
    replacement_capability: ReplacementCapability = ReplacementCapability.NONE


class PurposeRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    targets: tuple[RouteTarget, ...] = Field(min_length=1)
    requires_distinct_models: int = Field(default=1, ge=1)
