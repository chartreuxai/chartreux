from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from chartreux.app_server._subagent_records import (
    _accepted_saved_grants,
    _blocking_denylist,
    _grant_entries,
    _grant_wire,
)
from chartreux.app_server.protocol import (
    DangerousCommandAuthorityReadResponse,
    DangerousCommandDenylistBlockWire,
    DangerousCommandRevocationOutcome,
    DangerousCommandRevokeResponse,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config._dangerous_command_persistence import (
    read_saved_dangerous_command_grants,
)
from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.dangerous_commands import OPERATION_REGISTRY

if TYPE_CHECKING:
    from chartreux.app_server._sessions import SessionRuntime


class DangerousCommandAuthority:
    """Dangerous-command grant authority inspection for the registered sessions.

    Narrow collaborator of the session runtime registry: sessions are
    resolved through registry-supplied delegates (never snapshotted), the
    dangerous-command store is reached through the resolved loop, and the
    root orchestrator — the only saved-grant read/write path — through the
    root resolver. No state moves and no locks: none of the authority
    operations takes the registry lock, and no publication is added to any
    path.
    """

    def __init__(
        self,
        resolve_live_session: Callable[[str], SessionRuntime],
        resolve_session_loop: Callable[[str], AgentLoop | None],
        resolve_root: Callable[[], SessionRuntime | None],
    ) -> None:
        self._resolve_live_session = resolve_live_session
        self._resolve_session_loop = resolve_session_loop
        self._resolve_root = resolve_root

    async def save_dangerous_command_grant(
        self, *, session_id: str, operation_id: str, expected_revision: str
    ) -> ConfigSaveResult:
        """Persist one approved dangerous-command grant through the root orchestrator only.

        The calling session (root or child) is resolved through the registry,
        and the save is keyed by that registered session's cwd; a caller-supplied
        project path is never trusted. Child orchestrators never write: the save
        always targets the root orchestrator. Mid-turn safe: the orchestrator
        mutation lock is the only reservation, and nothing is published.
        """
        caller = self._resolve_session_loop(session_id)
        root_runtime = self._resolve_root()
        if caller is None or root_runtime is None:
            raise ValueError(
                "Dangerous-command grant save requires a registered session and root"
            )
        return await root_runtime.agent_loop.config_orchestrator.save_dangerous_command_grant(
            project=caller.cwd,
            operation_id=operation_id,
            expected_revision=expected_revision,
        )

    def _dangerous_command_grant_save(
        self, session_id: str, project: Path, operation_id: str, expected_revision: str
    ) -> Awaitable[ConfigSaveResult]:
        # The loop's proposal project is not a trusted key: the save is keyed
        # by the registered caller session's cwd, exactly like the roots save.
        del project
        return self.save_dangerous_command_grant(
            session_id=session_id,
            operation_id=operation_id,
            expected_revision=expected_revision,
        )

    async def read_dangerous_command_authority(
        self, session_id: str
    ) -> DangerousCommandAuthorityReadResponse:
        """Inspect one registered session's dangerous-command authority.

        The project is always the registered session's own cwd; no
        caller-supplied project key is accepted. Live grants come from the
        session's own store, accepted saved grants from the restrictions the
        session enforces, and the disk read is the fresh P2 projection with
        its revision. Blocking denylist entries are reported apart from grant
        authority because no grant overrides an explicit denylist veto.
        """
        runtime = self._resolve_live_session(session_id)
        loop = runtime.agent_loop
        store = loop._dangerous_command_grants
        project = loop.cwd
        root = self._resolve_root()
        if root is None:
            raise ValueError("Dangerous-command authority requires a registered root")
        saved = await read_saved_dangerous_command_grants(
            root.agent_loop.config_orchestrator, project=project
        )
        return DangerousCommandAuthorityReadResponse(
            project=str(project),
            live_authority_generation=store.authority_generation,
            live_grants=[
                _grant_wire(entry)
                for entry in _grant_entries(store.dangerous_command_authorities())
            ],
            accepted_saved_grants=[
                _grant_wire(entry) for entry in _accepted_saved_grants(loop, project)
            ],
            saved_operations=(
                list(saved.operations) if saved.unavailable is None else None
            ),
            disk_revision=saved.user_revision,
            blocking_denylist=[
                DangerousCommandDenylistBlockWire(
                    pattern=block.pattern, operations=list(block.operations)
                )
                for block in _blocking_denylist(loop)
            ],
        )

    async def revoke_dangerous_command_grant(
        self,
        *,
        session_id: str,
        operation_id: str,
        expected_live_generation: int,
        expected_disk_revision: str | None,
    ) -> DangerousCommandRevokeResponse:
        """Revoke one operation's live and saved authority for a session.

        Live removal is guarded by the session store's monotonic generation:
        a stale expected generation aborts the whole revocation (the saved
        side is untouched) and reports the current generation to re-read. A
        matching generation removes the operation's session grants at every
        scope and invalidates installed pending once permits immediately;
        cached authority is dropped. A question already awaiting an answer is
        not cancelled: answering it after the revocation still grants fresh
        authority for the decided scope. Saved removal is the dedicated
        revision-checked user-layer remove routed through the root
        orchestrator and keyed by the registered session's cwd. The two
        outcomes are reported separately, and a failed saved removal warns
        that the authority can return on the next reload or start. When the
        session's accepted saved restrictions still authorize the revoked
        operation — whether or not the runtime store ever held it — the
        removal warns that the session keeps that authority until the next
        reload or start. The reported generation is read after the saved-side
        attempt, so a concurrent approval during the disk write is
        reflected.
        """
        runtime = self._resolve_live_session(session_id)
        loop = runtime.agent_loop
        store = loop._dangerous_command_grants
        if store.authority_generation != expected_live_generation:
            return DangerousCommandRevokeResponse(
                live=DangerousCommandRevocationOutcome(
                    status="conflict", generation=store.authority_generation
                ),
                saved=DangerousCommandRevocationOutcome(status="skipped"),
                live_authority_generation=store.authority_generation,
            )
        requirements = {
            (operation_id, authority.scope)
            for authority in store.dangerous_command_authorities()
            if operation_id in authority.operations
        }
        store.revoke_session(requirements)
        store.invalidate_pending(requirements)
        loop.tool_manager._invalidate_dangerous_command_authority()
        operation = OPERATION_REGISTRY[operation_id]
        warnings: list[str] = []
        if any(
            entry.operation_id == operation_id
            for entry in _accepted_saved_grants(loop, loop.cwd)
        ):
            # The runtime store no longer carries the operation — it may
            # never have — but the session's accepted restrictions still
            # authorize it, so the session's live authority persists until a
            # reload or restart.
            warnings.append(
                f"The live {operation_id} grant is removed, but this session "
                f"keeps {operation_id} authority from accepted config until "
                "the next config reload or session start."
            )
        if not operation.persistable:
            # Consequential operations are never saved, so no saved authority
            # can exist to remove.
            saved_outcome = DangerousCommandRevocationOutcome(status="not_applicable")
        elif expected_disk_revision is None:
            saved_outcome = DangerousCommandRevocationOutcome(status="skipped")
        else:
            root = self._resolve_root()
            if root is None:
                raise ValueError(
                    "Dangerous-command revocation requires a registered root"
                )
            result = await root.agent_loop.config_orchestrator.remove_dangerous_command_grant(
                project=loop.cwd,
                operation_id=operation_id,
                expected_revision=expected_disk_revision,
            )
            if result.error == "conflict":
                saved_outcome = DangerousCommandRevocationOutcome(status="conflict")
            elif result.persistence == "saved" and result.error is None:
                saved_outcome = DangerousCommandRevocationOutcome(
                    status="removed", revision=result.revision
                )
            else:
                saved_outcome = DangerousCommandRevocationOutcome(
                    status="failed", revision=result.revision
                )
            if saved_outcome.status != "removed":
                warnings.append(
                    f"Removing the saved {operation_id} grant for {loop.cwd} did "
                    "not succeed, so the authority can return on the next config "
                    "reload or session start."
                )
        # Read the generation after the saved-side attempt: a concurrent
        # approval during the disk write bumps it, and the response reports
        # the post-operation value.
        generation = store.authority_generation
        return DangerousCommandRevokeResponse(
            live=DangerousCommandRevocationOutcome(
                status="removed", generation=generation
            ),
            saved=saved_outcome,
            live_authority_generation=generation,
            warning=" ".join(warnings) if warnings else None,
        )
