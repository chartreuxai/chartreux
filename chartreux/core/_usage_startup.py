"""Loop-independent, single-use accounting identity for pre-session inference."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
from threading import Lock

from chartreux.core._usage_io import reserve_usage_root
from chartreux.core.config import ModelConfig
from chartreux.core.session.session_id import generate_session_id
from chartreux.core.usage import (
    AsyncUsageWriter,
    CoverageWarning,
    CoverageWarningCode,
    UsageAttribution,
    UsagePurpose,
    UsageRecord,
    UsageWriter,
    UsageWriteResult,
)
from chartreux.core.usage_project import resolve_project_key_async


@dataclass(frozen=True, slots=True)
class StartupAccountingIdentity:
    root_session_id: str
    project_key: str
    # The store directory, as accepted by UsageWriter (not the root subdirectory).
    usage_dir: Path


class StartupAccountingContext:
    """Contains no loop-owned resources; each facade belongs to its caller's loop.

    Early producers must settle and drain their facade before closing their loop.
    A claim is terminal even if construction fails. Keep only content-free
    settlements here, so failed append warnings survive a CLI loop handoff.
    """

    def __init__(
        self,
        identity: StartupAccountingIdentity,
        *,
        warnings: tuple[CoverageWarning, ...] = (),
    ) -> None:
        self._identity = identity
        self._writer = UsageWriter(identity.usage_dir)
        self._lock = Lock()
        self._state = "available"
        self._early_settlements: tuple[tuple[UsageRecord, UsageWriteResult], ...] = ()
        self.warnings = warnings

    @property
    def identity(self) -> StartupAccountingIdentity:
        return self._identity

    @property
    def writer(self) -> UsageWriter:
        return self._writer

    @property
    def early_settlements(self) -> tuple[tuple[UsageRecord, UsageWriteResult], ...]:
        with self._lock:
            return self._early_settlements

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def claim(self) -> None:
        with self._lock:
            if self._state != "available":
                raise RuntimeError("Startup accounting context is single-use")
            self._state = "claimed"

    def commit_adoption(self) -> None:
        with self._lock:
            if self._state != "claimed":
                raise RuntimeError("Startup accounting context was not claimed")
            self._state = "adopted"

    def abandon(self) -> None:
        with self._lock:
            if self._state != "adopted":
                self._state = "abandoned"

    def on_early_settlement(
        self, record: UsageRecord, result: UsageWriteResult
    ) -> None:
        with self._lock:
            self._early_settlements += ((record, result),)

    def early_writer(self) -> AsyncUsageWriter:
        """Create a caller-owned facade; never retain it across loops."""
        with self._lock:
            if self._state != "available":
                raise RuntimeError("Startup accounting context is single-use")
        writer = AsyncUsageWriter(writer=self.writer)
        writer.subscribe(self.on_early_settlement)
        return writer

    def attribution(self, model: ModelConfig) -> UsageAttribution:
        return UsageAttribution(
            root_session_id=self.identity.root_session_id,
            session_id=self.identity.root_session_id,
            parent_session_id=None,
            agent_role="startup",
            agent_profile=None,
            purpose=UsagePurpose.WORKTREE_NAMING,
            model=model.alias,
            provider=model.provider,
            wire_name=model.name,
            project_key=self.identity.project_key,
        )


def _allocate(project_key: str, usage_dir: Path) -> StartupAccountingContext:
    for _ in range(32):
        root_id = generate_session_id()
        try:
            if not reserve_usage_root(usage_dir, root_id):
                continue
        except OSError:
            logging.getLogger(__name__).warning(
                "Usage ledger reservation failed; recorded usage coverage is degraded"
            )
            return StartupAccountingContext(
                StartupAccountingIdentity(root_id, project_key, usage_dir),
                warnings=(
                    CoverageWarning(
                        code=CoverageWarningCode.WRITE_FAILED, root_session_id=root_id
                    ),
                ),
            )
        return StartupAccountingContext(
            StartupAccountingIdentity(root_id, project_key, usage_dir)
        )
    raise RuntimeError("Could not reserve a unique startup accounting identity")


async def create_startup_accounting_context(
    original_workspace: Path, *, usage_dir: Path | None = None
) -> StartupAccountingContext:
    """Resolve the original project and exclusively reserve before naming."""
    from chartreux.core.paths import USAGE_DIR

    project_key = await resolve_project_key_async(original_workspace)
    directory = usage_dir if usage_dir is not None else USAGE_DIR.path
    return await asyncio.to_thread(_allocate, project_key, directory)
