"""Production root-policy storage surfaces shared by the [6B] differential traces.

The oracle and the competing-writer probe both drive the real revision-checked
user-layer writer (:func:`chartreux.core.config._root_persistence.persist_project_root_grant`):
the persistence entry point reads exactly one installed actual user layer from
the orchestrator it is handed, so both callers provide that layer over a test
``tmp_path`` store instead of a running session's config tree. Nothing here
re-implements the merge, the validation, the revision check, or the write.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from chartreux.core.config._root_persistence import persist_project_root_grant
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.types import ConfigSaveResult


class OrchestratorHost:
    """The one attribute root persistence reads: the installed layer list."""

    def __init__(self, layer: UserConfigLayer) -> None:
        self.layers = [layer]


async def read_revision(config: Path) -> str:
    """Read the store's current revision through the production layer."""
    layer = UserConfigLayer(path=config)
    await layer.load(force=True)
    revision = layer.fingerprint
    if revision is None:
        raise RuntimeError(f"no revision for {config}")
    return revision


async def write_project_root(
    *, config: Path, project: Path, root: Path, expected_revision: str
) -> ConfigSaveResult:
    """Persist one project root grant through the production writer."""
    return await persist_project_root_grant(
        cast("Any", OrchestratorHost(UserConfigLayer(path=config))),
        project=project,
        root=root,
        expected_revision=expected_revision,
    )
