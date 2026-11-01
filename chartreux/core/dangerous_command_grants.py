"""Runtime-owned persistence port for dangerous-command project grants.

The agent loop reports an approved "Always for this project" grant through
this port; a later package wires the port to the protected config save. The
port reports the persistence outcome honestly and separately from the live
session grant, which the loop applies itself regardless of the persistence
outcome. This module defines the port only: no grant mutation, no config IO.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from chartreux.core.config.types import ConfigSaveResult


class DangerousCommandGrantPersistencePort(Protocol):
    """Runtime-owned port the loop reports approved project grants through."""

    async def save_dangerous_command_grant(
        self, session_id: str, project: Path, operation_id: str, expected_revision: str
    ) -> ConfigSaveResult:
        """Persist one approved grant for the calling session's project.

        The runtime routes the save to the grant orchestrator keyed by the
        registered caller session's project; child orchestrators never write.
        The result reports persistence honestly, including conflicts, write
        failures, and durability uncertainty.
        """
        ...

    async def save_remote_exec_grant(
        self, session_id: str, project: Path, destination: str, expected_revision: str
    ) -> ConfigSaveResult:
        """Persist one approved remote-exec destination for the session's project.

        Same routing and honors as the operation grant: the save goes to the
        grant orchestrator keyed by the registered caller session's project,
        and the result reports persistence honestly.
        """
        ...


class NoOpDangerousCommandGrantPersistence:
    """Default port: this runtime persists no dangerous-command grants."""

    async def save_dangerous_command_grant(
        self, session_id: str, project: Path, operation_id: str, expected_revision: str
    ) -> ConfigSaveResult:
        return ConfigSaveResult("user", "not_saved", "unchanged")

    async def save_remote_exec_grant(
        self, session_id: str, project: Path, destination: str, expected_revision: str
    ) -> ConfigSaveResult:
        return ConfigSaveResult("user", "not_saved", "unchanged")
