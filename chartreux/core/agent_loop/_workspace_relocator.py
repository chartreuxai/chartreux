"""Workspace-relocation collaborator for AgentLoop (ALL cluster 6, WorkspaceRelocator).

Owns the relocation transaction and the trust a move acquired: the
reservation that keeps other operations out across the move's awaits,
the destination validation that restricts targets to checkouts of the
session's own repository, the bind/restore pair that points the loop at
the new workspace, and the tracking of the session trust a move granted
so a later move releases that grant and not one somebody else made.

The loop retains the workspace roots themselves (``cwd`` and
``harness_files``, reached through the late-bound ``assign_workspace``
seam), the session holder gate, the active-turn check, and the
consumer-rebuild facade (``reload_with_initial_messages``, which the
configuration transactions drive). Live loop values — the workspace
roots, the agent manager, the config orchestrator, and the session
logger — are read through getters, never captured at construction.

Binding contracts preserved by the move:

- The transaction is never split: reservation, destination validation,
  bind, and the rollback that revokes only the trust the move acquired
  run as one operation on this service.
- Rollback revokes ONLY the move-acquired trust: the trust store is
  process-wide and counts grants, so releasing the departure outright
  would take back a grant the session never made — one made at session
  start, or one another session sitting there holds.
- A superseded reload is NOT a failed relocation: the consumer rebuild
  returns without committing when a newer reload supersedes it, and
  treating supersession as failure would roll back against a reload that
  is legitimately in flight.
- The clear-denylist-but-not-root-grants asymmetry is PINNED, not fixed:
  reset/rebind clears dangerous-command state but does NOT clear root
  grants, and ``ToolManager.reset_all()`` only clears instances. That
  asymmetry lives in the session transitions and is preserved exactly;
  any policy correction needs separate authorization.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import contextlib
from pathlib import Path

from chartreux.core.agent_loop.errors import AgentLoopStateError
from chartreux.core.agents.manager import AgentManager
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.git.errors import GitError
from chartreux.core.git.worktree.repository import WorktreeRepository
from chartreux.core.session.session_logger import SessionLogger
from chartreux.observability.logging import logger


class WorkspaceRelocator:
    """Own the relocation transaction and the trust a move acquired.

    ``relocate`` refuses while a turn is active or another operation holds
    the session, validates the destination as a checkout of the session's
    own repository, grants the destination session trust, binds the new
    workspace, and on any failure of the bind revokes exactly the trust
    the move granted and restores the previous workspace. The session
    holder is taken before the first await and held across both, so an
    operation cannot start against a directory that is moving under it.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        cwd_getter: Callable[[], Path],
        harness_files_getter: Callable[[], HarnessFilesManager],
        agent_manager_getter: Callable[[], AgentManager],
        config_orchestrator_getter: Callable[
            [], ConfigOrchestrator[ChartreuxConfigSchema]
        ],
        session_logger_getter: Callable[[], SessionLogger],
        has_active_turn: Callable[[], bool],
        require_background_jobs_idle: Callable[[], None],
        take_session: Callable[[str], None],
        release_session: Callable[[str], None],
        assign_workspace: Callable[[Path, HarnessFilesManager], None],
        destination_checkout: Callable[[Path], Path | None],
        bind_workspace: Callable[[Path, HarnessFilesManager], Awaitable[None]],
        reload_with_initial_messages: Callable[..., Awaitable[None]],
    ) -> None:
        # Live getters: the workspace roots are replaced by the bind/restore
        # themselves, the agent manager and config orchestrator are rebuilt by
        # configuration transactions, and the session logger is constructed
        # after this service — all resolved at call time, never captured.
        self._cwd_getter = cwd_getter
        self._harness_files_getter = harness_files_getter
        self._agent_manager_getter = agent_manager_getter
        self._config_orchestrator_getter = config_orchestrator_getter
        self._session_logger_getter = session_logger_getter
        self._has_active_turn = has_active_turn
        # Late-bound loop seams: the holder gate and the background-jobs idle
        # check are the loop's own operations, the workspace assignment is the
        # loop's own state mutation, and the destination validation, the bind,
        # and the consumer rebuild resolve the loop surfaces at call time, so
        # instance patches keep intercepting (the test doubles patch
        # _destination_checkout, _bind_workspace, and — through the reload
        # facade — _prepare_reload).
        self._require_background_jobs_idle = require_background_jobs_idle
        self._take_session = take_session
        self._release_session = release_session
        self._assign_workspace = assign_workspace
        self._destination_checkout = destination_checkout
        self._bind_workspace = bind_workspace
        self._reload_with_initial_messages = reload_with_initial_messages
        # Move-acquired trust tracking: the directory a move granted session
        # trust to, so a later move releases that and not a grant somebody
        # else made.
        self._trust_taken_by_move: Path | None = None

    async def relocate(self, cwd: Path) -> None:
        """Move the session to *cwd*, or leave it exactly where it was.

        Refused while a turn is active. Tool calls in a batch run as concurrent
        tasks, so a move alongside a file or shell call would leave that call
        writing to the old directory, and swapping the objects it holds cannot
        stop one already running. A subagent runs inside its parent's turn, so
        this covers those too.

        Refused as well while an operation holds the session. A turn is not
        the only thing that reads the working directory or acts on its
        repository, and the rest are invisible to the turn check.

        Raises:
            AgentLoopStateError: Something is in flight, or *cwd* is not a
                directory.
        """
        if self._has_active_turn():
            raise AgentLoopStateError("Cannot relocate while a turn is active")
        target = cwd.expanduser().resolve()
        if target == self._cwd_getter():
            return
        self._require_background_jobs_idle()
        if not target.is_dir():
            raise AgentLoopStateError(f"Not a directory: {target}")

        # Taken before the first await, and held across both. Checking once and
        # then awaiting would leave a window for an operation to start against a
        # directory that is moving under it.
        self._take_session("relocation")
        try:
            checkout = await asyncio.to_thread(self._destination_checkout, target)
            if checkout is None:
                raise AgentLoopStateError(
                    f"Not a worktree of this session's repository: {target}"
                )

            previous_files = self._harness_files_getter()
            previous_cwd = self._cwd_getter()
            moved = self._harness_files_getter().moved_to(target)
            # The checkout rather than the position inside it. The project layer
            # finds `.chartreux` by walking up from the working directory and asks
            # whether the directory holding it is trusted, and trust resolves
            # upward too - so a grant on a subdirectory never reaches the root
            # the file sits at, and a subdirectory move would read no project
            # config at all. That is the case re-rooting exists for.
            #
            # Safe only because of the check above. Session trust outranks an
            # explicit untrust at the same path and reaches every descendant,
            # so granting it for an arbitrary directory would hand the session
            # a tree the user never opened. Restricted to a checkout of the
            # repository it is already working in, it inherits authority rather
            # than creating it.
            moved.trust_store.trust_for_session(checkout)
            try:
                await self._bind_workspace(target, moved)
            except BaseException:
                moved.trust_store.revoke_session_trust(checkout)
                await self._restore_workspace(previous_cwd, previous_files)
                raise
            # Release the grant a move took, and only that one. The trust store
            # is process-wide and counts grants, so releasing the departure
            # outright would take back one the session never made: at session
            # start for this session, or for another session sitting there.
            if self._trust_taken_by_move is not None:
                moved.trust_store.revoke_session_trust(self._trust_taken_by_move)
            self._trust_taken_by_move = checkout
        finally:
            self._release_session("relocation")

    def release_move_trust(self) -> None:
        """Revoke only the trust this session's own moves acquired.

        The close path: the trust store is process-wide and counts grants,
        so the move-acquired grant is released and nothing else; the loop
        then revokes every remaining session trust itself.
        """
        if self._trust_taken_by_move is not None:
            self._harness_files_getter().trust_store.revoke_session_trust(
                self._trust_taken_by_move
            )
            self._trust_taken_by_move = None

    def destination_checkout(self, target: Path) -> Path | None:
        """The checkout *target* sits in, or None when it is not one to move to.

        The precondition the rest of the move rests on, and it has to be exactly
        this narrow. Anything wider grants trust over a tree the session never
        occupied; anything narrower strands a session opened in a subdirectory,
        since it moves into the matching subdirectory of the worktree and could
        not name its way back.

        So the destinations are the counterparts of the current position: the
        same relative path in the main checkout, and in each linked worktree.
        Not the parent checkout, which would widen a subdirectory session to the
        whole repository, and not a sibling directory, which the session has no
        claim on either.

        The checkout comes back with the answer because trust is granted on it
        rather than on the position: a `.chartreux` at the root is out of reach of a
        grant made on a subdirectory below it.
        """
        try:
            with WorktreeRepository.open(self._cwd_getter()) as repository:
                for worktree in repository.linked():
                    if worktree.path.resolve() == target:
                        return worktree.root.resolve()
                if repository.repository_counterpart == target:
                    return repository.root
        except GitError:
            return None
        return None

    def _project_config_layer(self) -> ProjectConfigLayer | None:
        """The live project config layer, or None when there is none to re-root.

        Absent when the project source is disabled, and when the orchestrator
        carries no layer stack at all, which is how the test doubles are built.
        """
        with contextlib.suppress(AttributeError, KeyError):
            layer = self._config_orchestrator_getter().get_layer(
                ProjectConfigLayer.NAME
            )
            if isinstance(layer, ProjectConfigLayer):
                return layer
        return None

    async def bind_workspace(
        self, cwd: Path, harness_files: HarnessFilesManager
    ) -> None:
        """Point everything rooted in the working directory at *cwd*."""
        self._assign_workspace(cwd, harness_files)
        # The logger holds its own copy and is what writes the directory into
        # session metadata, so a move that skipped it would be recorded as
        # never having happened.
        self._session_logger_getter().relocated_to(cwd)
        if (layer := self._project_config_layer()) is not None:
            await layer.reroot(cwd)
            await self._config_orchestrator_getter().reload()
        # Rebuilds the tools, skills, prompt and hooks bound to the old
        # directory. Preparing happens off-thread and the commit is synchronous,
        # so that half is not observable partly applied. It runs last so those
        # objects are built from the config the destination resolves to.
        #
        # It returns without committing, and without raising, when a newer
        # reload supersedes it. The destination still wins: that newer reload
        # prepares from the live cwd, which is already the new one. What is
        # not guaranteed is that the rebuild has finished by the time relocate
        # returns. Treating supersession as failure would be worse, since the
        # rollback would then fight a reload that is legitimately in flight.
        await self._reload_with_initial_messages(reload_hooks=True)

    async def _restore_workspace(
        self, cwd: Path, harness_files: HarnessFilesManager
    ) -> None:
        """Undo a partly applied move.

        Failures are logged rather than raised. The caller is already unwinding
        the exception that says why the move did not happen, and replacing it
        with a second one would hide that.
        """
        self._assign_workspace(cwd, harness_files)
        self._session_logger_getter().relocated_to(cwd)
        try:
            if (layer := self._project_config_layer()) is not None:
                await layer.reroot(cwd)
                await self._config_orchestrator_getter().reload()
        except Exception:
            logger.exception(
                "Failed to restore the workspace after a rejected move cwd=%s", cwd
            )
