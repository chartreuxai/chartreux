"""Protocol-only Shell Permissions panel service boundary.

The panel inspects and revokes one registered session's dangerous-command
authority through the P5 client resource methods. This module owns no config
writes, no authority decisions, and no persistence behavior: it maps the wire
inspection and revocation responses into immutable panel state so the widget
renders live session grants, saved project grants (with the disk revision),
and blocking denylist entries as separate, honest sections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from chartreux.app_server.protocol import (
    DangerousCommandAuthorityReadResponse,
    DangerousCommandRevokeResponse,
)

ShellPermissionsGrantSource = Literal["live", "saved", "saved_disk"]
ShellPermissionsSideStatus = Literal[
    "removed", "conflict", "skipped", "not_applicable", "failed"
]


class ShellPermissionsResource(Protocol):
    """The client resource surface the panel drives (P5 endpoints)."""

    async def read_dangerous_command_authority(
        self,
    ) -> DangerousCommandAuthorityReadResponse: ...

    async def revoke_dangerous_command_grant(
        self,
        *,
        operation_id: str,
        expected_live_generation: int,
        expected_disk_revision: str | None,
        user_initiated: bool,
    ) -> DangerousCommandRevokeResponse: ...


@dataclass(frozen=True)
class ShellPermissionsGrant:
    """One dangerous-command authority entry: an operation in one scope."""

    operation_id: str
    repository: str
    source: ShellPermissionsGrantSource


@dataclass(frozen=True)
class ShellPermissionsDenylistBlock:
    """An explicit shell denylist entry vetoing dangerous git operations."""

    pattern: str
    operations: tuple[str, ...]


@dataclass(frozen=True)
class ShellPermissionsInspection:
    """The immutable panel state projected from one authority read.

    ``live_grants`` is the runtime-owned authority of the session's own store.
    ``saved_grants`` is the saved user-config authority the session's accepted
    restrictions contribute. ``disk_only_operations`` are operations present on
    disk but not accepted by the running session, reported apart so the saved
    section stays truthful about what a revocation can remove. The disk revision
    is the expected revision for a saved removal; ``saved_operations_unavailable``
    marks a disk read that could not be projected.
    """

    project: str
    live_authority_generation: int
    disk_revision: str | None
    live_grants: tuple[ShellPermissionsGrant, ...]
    saved_grants: tuple[ShellPermissionsGrant, ...]
    disk_only_operations: tuple[str, ...]
    saved_operations_unavailable: bool
    blocking_denylist: tuple[ShellPermissionsDenylistBlock, ...]


@dataclass(frozen=True)
class ShellPermissionsRevocation:
    """One user-initiated revocation, live and saved sides reported apart."""

    operation_id: str
    live_status: ShellPermissionsSideStatus
    saved_status: ShellPermissionsSideStatus
    live_generation: int | None
    live_authority_generation: int
    saved_revision: str | None
    warning: str | None

    @property
    def conflicted(self) -> bool:
        """A stale expected revision rejected the removal on either side."""
        return "conflict" in {self.live_status, self.saved_status}


def _repository_display(work_tree: str | None, git_dir: str) -> str:
    return work_tree or git_dir


def _grant(
    operation_id: str,
    work_tree: str | None,
    git_dir: str,
    source: ShellPermissionsGrantSource,
) -> ShellPermissionsGrant:
    return ShellPermissionsGrant(
        operation_id=operation_id,
        repository=_repository_display(work_tree, git_dir),
        source=source,
    )


class ShellPermissionsService:
    """Inspect and revoke dangerous-command authority through the resource."""

    def __init__(self, resource: ShellPermissionsResource) -> None:
        self._resource = resource

    async def inspect(self) -> ShellPermissionsInspection:
        """Read the session's live, accepted-saved, and disk authority."""
        response = await self._resource.read_dangerous_command_authority()
        accepted = {grant.operation_id for grant in response.accepted_saved_grants}
        disk_only = (
            tuple(
                operation_id
                for operation_id in response.saved_operations or ()
                if operation_id not in accepted
            )
            if response.saved_operations is not None
            else ()
        )
        return ShellPermissionsInspection(
            project=response.project,
            live_authority_generation=response.live_authority_generation,
            disk_revision=response.disk_revision,
            live_grants=tuple(
                _grant(
                    grant.operation_id,
                    grant.scope.work_tree,
                    grant.scope.git_dir,
                    "live",
                )
                for grant in response.live_grants
            ),
            saved_grants=tuple(
                _grant(
                    grant.operation_id,
                    grant.scope.work_tree,
                    grant.scope.git_dir,
                    "saved",
                )
                for grant in response.accepted_saved_grants
            ),
            disk_only_operations=disk_only,
            saved_operations_unavailable=response.saved_operations is None,
            blocking_denylist=tuple(
                ShellPermissionsDenylistBlock(
                    pattern=block.pattern, operations=tuple(block.operations)
                )
                for block in response.blocking_denylist
            ),
        )

    async def revoke(
        self,
        *,
        operation_id: str,
        expected_live_generation: int,
        expected_disk_revision: str | None,
    ) -> ShellPermissionsRevocation:
        """Revoke one operation's live and saved authority (user-initiated).

        The expected live generation and disk revision come from the last
        inspection; a stale expectation is a reported conflict outcome, never
        an exception. Endpoint errors propagate so the panel can surface them.
        """
        response = await self._resource.revoke_dangerous_command_grant(
            operation_id=operation_id,
            expected_live_generation=expected_live_generation,
            expected_disk_revision=expected_disk_revision,
            user_initiated=True,
        )
        return ShellPermissionsRevocation(
            operation_id=operation_id,
            live_status=response.live.status,
            saved_status=response.saved.status,
            live_generation=response.live.generation,
            live_authority_generation=response.live_authority_generation,
            saved_revision=response.saved.revision,
            warning=response.warning,
        )
