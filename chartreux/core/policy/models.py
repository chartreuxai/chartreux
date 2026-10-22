from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ResourceKind(StrEnum):
    IDENTITY = "identity"
    WORKFLOW = "workflow"
    STRATEGY = "strategy"
    OVERLAY = "overlay"


@dataclass(frozen=True)
class ResolvedResource:
    kind: ResourceKind
    id: str
    source: str
    content: str | None
    path: Path | None = None
