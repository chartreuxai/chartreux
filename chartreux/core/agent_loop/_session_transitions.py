"""Session-transition collaborator for AgentLoop (ALL cluster 6, steps 1-4).

Owns the session-transition preparation extracted from _loop.py: the
todo/file-state replay participants, the active-model pricing helper, the
resume preparation, the reset preparation with its lease cleanup, and the
clear-history orchestration. The resume's synchronous commit stays on the
loop (ALL §6: "resume has a synchronous commit"); the reset's commit stays on
the loop for the same reason. The service owns the temporary prepared values
and the cleanup obligation of retaining the departing title producer before
the reset drops its reference.

The loop retains session identity, generation, stats, the accounting owner,
and lease publication. Borrowed shared ownership (never cloned or replaced):
the file-state context, the message list, the out-of-band event queue, and the
detached accounting-producer set. Live loop values (config, managers, stats,
the session identity and lease, the bound policy, the session logger, the
middleware pipeline, the scratchpad, the accounting owner, the title
producer) are read through getters, so manager replacement and stats
replacement stay visible. The scratchpad preparation is reached through the
loop's ``prepare_scratchpad_for_session`` seam (it reads the patched
module-level ``init_scratchpad`` surface, a stage-13 patch site), and the
loop's ``require_background_jobs_idle``/``_render_system_prompt``/
``_reset_session`` are reached through late-bound callbacks, so instance
patches keep intercepting. Session-ID minting resolves the patched
module-level ``generate_session_id`` surface through a facade callback, and
the session logger is read through a getter because it is constructed after
this service and instance-patched by tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from chartreux.core.agent_loop._file_state_context import FileStateContext
from chartreux.core.agent_loop._loop_types import RootAccountingOwner
from chartreux.core.agents.manager import AgentManager
from chartreux.core.background_jobs import BackgroundJobRegistry, BackgroundJobsPort
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.dispatch.renderer import task_description_for_config
from chartreux.core.dispatch.session import (
    BoundDispatchPolicy,
    bind_policy,
    resume_policy,
)
from chartreux.core.events import BaseEvent
from chartreux.core.file_state import FileStateStore
from chartreux.core.llm_models import LLMMessage
from chartreux.core.message_list import MessageList
from chartreux.core.middleware import MiddlewarePipeline
from chartreux.core.session.session_id import extract_suffix
from chartreux.core.session.session_lease import SessionLease
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session_types import AgentStats, SessionMetadata
from chartreux.core.skills.manager import SkillManager
from chartreux.core.tools.builtins.todo import TodoState
from chartreux.core.tools.manager import NoSuchToolError, ToolManager
from chartreux.observability.logging import logger


class _PrepareScratchpad:
    pass


_PREPARE_SCRATCHPAD = _PrepareScratchpad()


@dataclass(frozen=True, slots=True)
class _PreparedResume:
    """Temporary values computed by resume preparation, applied by the loop's commit."""

    resumed_policy: BoundDispatchPolicy
    replacement_jobs: BackgroundJobRegistry | None
    replacement_port: BackgroundJobsPort | None
    owner: RootAccountingOwner | None
    previous_scratchpad: Path | None
    scratchpad_dir: Path | None
    resumed_task_description: str
    resumed_prompt: str


@dataclass(frozen=True, slots=True)
class _PreparedReset:
    """Temporary values computed by reset preparation, applied by the loop's commit."""

    session_id: str
    parent_session_id: str | None
    new_policy: BoundDispatchPolicy
    new_task_description: str
    new_prompt: str
    owner: RootAccountingOwner | None
    lease: SessionLease | None


class SessionTransitions:
    """Own the session-transition replay participants and transition preparation.

    ``restore_todo_state``/``restore_file_state`` replay the resumed
    transcript's todo and working-set state; ``apply_active_model_pricing``
    rebinds the fresh stats to the active model's prices;
    ``prepare_resume`` computes every value a resume rebind needs without
    mutating the live loop and returns them as a ``_PreparedResume`` for the
    loop's synchronous commit to apply; ``prepare_reset`` computes every value
    a reset needs (fresh identity, policy, prompt, accounting owner, lease)
    and releases the lease on any failure, returning them as a
    ``_PreparedReset`` for the loop's commit; ``clear_history`` orchestrates
    the distinct history-clear protocol: save, reset, trim, fresh stats.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        file_state_context: FileStateContext,
        out_of_band_events: asyncio.Queue[BaseEvent],
        detached_accounting_producers: set[asyncio.Task[None]],
        config_getter: Callable[[], ChartreuxConfigSchema],
        agent_manager_getter: Callable[[], AgentManager],
        tool_manager_getter: Callable[[], ToolManager],
        skill_manager_getter: Callable[[], SkillManager],
        stats_getter: Callable[[], AgentStats],
        is_subagent_getter: Callable[[], bool],
        background_jobs_getter: Callable[[], BackgroundJobsPort | None],
        job_lifetime_generation_getter: Callable[[], int],
        scratchpad_dir_getter: Callable[[], Path | None],
        accounting_owner_getter: Callable[[], RootAccountingOwner | None],
        auto_title_task_getter: Callable[[], asyncio.Task[None] | None],
        require_background_jobs_idle: Callable[[], None],
        prepare_scratchpad: Callable[[str], Path | None],
        render_system_prompt: Callable[
            [
                SkillManager,
                ChartreuxConfigSchema,
                BoundDispatchPolicy | None,
                Path | None | _PrepareScratchpad,
            ],
            str,
        ],
        messages: MessageList,
        session_id_getter: Callable[[], str],
        parent_session_id_getter: Callable[[], str | None],
        session_lease_getter: Callable[[], SessionLease | None],
        bound_dispatch_policy_getter: Callable[[], BoundDispatchPolicy],
        session_logger_getter: Callable[[], SessionLogger],
        middleware_pipeline_getter: Callable[[], MiddlewarePipeline],
        reset_session: Callable[[bool], Awaitable[None]],
        publish_fresh_stats: Callable[[], None],
        generate_session_id: Callable[[str | None], str],
    ) -> None:
        # Borrowed shared ownership: the file-state context, the message list,
        # the out-of-band event queue, and the detached accounting-producer set
        # are the loop's own objects, never cloned or replaced (the title-
        # producer retention registers into the same set the loop and the
        # runtime read; the clear orchestration trims the same list the loop
        # reads).
        self._file_state_context = file_state_context
        self._messages = messages
        self._out_of_band_events = out_of_band_events
        self._detached_accounting_producers = detached_accounting_producers
        # Live getters: the config is re-read per operation (the dispatch
        # policy rides it), the tool/agent managers and the stats object are
        # replaced by configuration transactions and session transitions, and
        # the scratchpad/accounting owner/title producer rotate across
        # transitions — all resolved at call time, never captured. The session
        # identity, lease, and bound policy ride the same treatment, and the
        # session logger (constructed after this service) and the middleware
        # pipeline (rebuilt by configuration transactions) are read through
        # getters so the current objects stay visible.
        self._config_getter = config_getter
        self._agent_manager_getter = agent_manager_getter
        self._tool_manager_getter = tool_manager_getter
        self._skill_manager_getter = skill_manager_getter
        self._stats_getter = stats_getter
        self._is_subagent_getter = is_subagent_getter
        self._background_jobs_getter = background_jobs_getter
        self._job_lifetime_generation_getter = job_lifetime_generation_getter
        self._scratchpad_dir_getter = scratchpad_dir_getter
        self._accounting_owner_getter = accounting_owner_getter
        self._auto_title_task_getter = auto_title_task_getter
        self._session_id_getter = session_id_getter
        self._parent_session_id_getter = parent_session_id_getter
        self._session_lease_getter = session_lease_getter
        self._bound_dispatch_policy_getter = bound_dispatch_policy_getter
        self._session_logger_getter = session_logger_getter
        self._middleware_pipeline_getter = middleware_pipeline_getter
        # Late-bound callbacks: the scratchpad preparation resolves the
        # patched module-level init_scratchpad surface through the loop seam
        # (test patches of the loop method keep intercepting), and the
        # background-jobs idle gate and the system-prompt render resolve the
        # loop's instance methods at call time. The reset callback is the
        # loop's own ``_reset_session`` (tests call and patch it directly, and
        # the RewindManager captures the bound method), the stats publication
        # stays loop-owned, and the session-ID facade resolves the patched
        # module-level generate_session_id surface at call time.
        self._require_background_jobs_idle = require_background_jobs_idle
        self._prepare_scratchpad = prepare_scratchpad
        self._render_system_prompt = render_system_prompt
        self._reset_session = reset_session
        self._publish_fresh_stats = publish_fresh_stats
        self._generate_session_id = generate_session_id

    def restore_todo_state(self, messages: Sequence[LLMMessage]) -> None:
        try:
            todo = self._tool_manager_getter().get("todo")
        except NoSuchToolError:
            return
        todo.state = TodoState.replay(messages)

    def restore_file_state(self, messages: Sequence[LLMMessage]) -> None:
        """Restore the working-set file state from the resumed transcript.

        Structural replay across compaction, mirroring ``_restore_todo_state``:
        the last compaction boundary's envelope wins, merged into the live
        store (the registry breaks ties between the live and replayed views).
        Fail-safe: a corrupt envelope degrades to the live state.
        """
        try:
            self._file_state_context.merge_replayed(FileStateStore.replay(messages))
        except Exception as exc:
            logger.warning("File state restore failed error=%s", type(exc).__name__)

    def apply_active_model_pricing(self) -> None:
        try:
            active_model = self._config_getter().get_active_model()
        except ValueError:
            return
        self._stats_getter().update_pricing(
            active_model.input_price,
            active_model.output_price,
            active_model.cached_input_price,
        )

    def prepare_resume(
        self,
        session_id: str,
        *,
        session_metadata: SessionMetadata,
        prepared_scratchpad: Path | None | _PrepareScratchpad,
        accounting_owner: RootAccountingOwner | None,
    ) -> _PreparedResume:
        """Prepare a resume rebind without mutating the live loop.

        The only intentionally fail-able work (scratchpad creation) runs
        before any mutation, so a failed preparation leaves the loop bound to
        its original session. The caller applies the returned record in the
        synchronous commit section, which stays on the loop.
        """
        self._require_background_jobs_idle()
        resumed_policy = resume_policy(
            self._config_getter(),
            session_metadata.dispatch_policy,
            self._agent_manager_getter().available_agents,
        )
        replacement_jobs = (
            BackgroundJobRegistry(
                generation=self._job_lifetime_generation_getter() + 1,
                event_sink=self._out_of_band_events.put_nowait,
            )
            if not self._is_subagent_getter()
            else None
        )
        # One getter read bound to a local: pyright does not narrow across two
        # call results (the original read the attribute twice).
        background_jobs = self._background_jobs_getter()
        replacement_port = (
            replacement_jobs.root_port()
            if replacement_jobs is not None
            else background_jobs.borrow()
            if background_jobs is not None
            else None
        )
        # Prepare without mutating the live loop. Enabled root accounting must
        # have its replacement allocated by the runtime before this sync boundary.
        departing_owner = self._accounting_owner_getter()
        owner = accounting_owner or departing_owner
        if owner is not None:
            if not self._is_subagent_getter() and owner.root_session_id != session_id:
                raise ValueError("Prepare accounting ownership before rebinding a root")
            if (
                departing_owner is not None
                and owner.root_session_id == departing_owner.root_session_id
            ):
                owner = departing_owner
        previous_scratchpad = self._scratchpad_dir_getter()
        scratchpad_dir = (
            self._prepare_scratchpad(session_id)
            if isinstance(prepared_scratchpad, _PrepareScratchpad)
            else prepared_scratchpad
        )

        # A child borrows access through ToolManager, never directory ownership.
        if self._is_subagent_getter() and scratchpad_dir is not None:
            raise ValueError("Child sessions cannot own a prepared scratchpad")

        resumed_config = (
            self._config_getter().model_copy().attach_dispatch_policy(resumed_policy)
        )
        resumed_task_description = task_description_for_config(resumed_config)
        resumed_prompt = self._render_system_prompt(
            self._skill_manager_getter(), resumed_config, resumed_policy, scratchpad_dir
        )

        # Retain title work before reset drops its reference. Cancellation is not
        # settlement: its frozen sink may still submit after this rebind. The old
        # writer stays process-owned; retirement requires producer settlement.
        title_task = self._auto_title_task_getter()
        if self._accounting_owner_getter() is not None and title_task is not None:
            self._detached_accounting_producers.add(title_task)
            title_task.add_done_callback(self._detached_accounting_producers.discard)

        return _PreparedResume(
            resumed_policy=resumed_policy,
            replacement_jobs=replacement_jobs,
            replacement_port=replacement_port,
            owner=owner,
            previous_scratchpad=previous_scratchpad,
            scratchpad_dir=scratchpad_dir,
            resumed_task_description=resumed_task_description,
            resumed_prompt=resumed_prompt,
        )

    async def prepare_reset(self, keep_parent: bool) -> _PreparedReset:
        """Prepare a session reset without mutating the live loop.

        The lease is acquired and the logger rebound inside one guarded
        section: any failure releases the lease and leaves the loop bound to
        its original session. The caller applies the returned record in the
        synchronous commit section, which stays on the loop. The reset keeps
        the loop's scratchpad (the render falls back to it), so unlike a
        resume it prepares no scratchpad of its own.
        """
        old_session_id = self._session_id_getter()
        suffix = extract_suffix(old_session_id)
        session_id = self._generate_session_id(suffix)
        session_lease = self._session_lease_getter()
        lease_root = (
            session_lease.path.parent.parent
            if session_lease is not None
            else Path(self._config_getter().session_logging.save_dir)
        )
        lease = (
            await asyncio.to_thread(SessionLease(lease_root, session_id).acquire)
            if self._config_getter().session_logging.enabled
            else None
        )
        is_subagent = self._is_subagent_getter()
        parent_session_id = (
            self._parent_session_id_getter()
            if keep_parent and is_subagent
            else old_session_id
            if keep_parent
            else None
        )
        try:
            new_policy = self._bound_dispatch_policy_getter()
            if not keep_parent and not is_subagent:
                new_policy = bind_policy(self._config_getter())
            new_config = (
                self._config_getter().model_copy().attach_dispatch_policy(new_policy)
            )
            new_task_description = task_description_for_config(new_config)
            new_prompt = self._render_system_prompt(
                self._skill_manager_getter(),
                new_config,
                new_policy,
                _PREPARE_SCRATCHPAD,
            )
            # One getter read bound to a local: no await runs between this read
            # and the title retention below (the except re-raises first on
            # failure), so the departing owner is read exactly once.
            departing_owner = self._accounting_owner_getter()
            owner = departing_owner
            if (
                owner is not None
                and not is_subagent
                and owner.root_session_id != session_id
            ):
                if owner.owner_factory is None:
                    raise ValueError(
                        "Prepare accounting ownership before resetting a root"
                    )
                owner = owner.owner_factory(session_id, owner.project_key)
            self._session_logger_getter().reset_session(
                session_id, parent_session_id=parent_session_id
            )
        except BaseException:
            if lease is not None:
                await asyncio.to_thread(lease.release)
            raise
        # Old invocation resources remain bound to the old process-owned writer.
        # Retain cancelled title producers just as on resume/rebind.
        title_task = self._auto_title_task_getter()
        if departing_owner is not None and title_task is not None:
            self._detached_accounting_producers.add(title_task)
            title_task.add_done_callback(self._detached_accounting_producers.discard)
        return _PreparedReset(
            session_id=session_id,
            parent_session_id=parent_session_id,
            new_policy=new_policy,
            new_task_description=new_task_description,
            new_prompt=new_prompt,
            owner=owner,
            lease=lease,
        )

    async def clear_history(self) -> None:
        """Run the distinct history-clear protocol: save, reset, trim, republish.

        Distinct from the rewind fork (a reset that keeps the parent) and
        from resume (an in-place rebind onto a loaded session): a clear saves
        the interaction, resets to a fresh parentless root, trims the live
        history to its system prompt, and republishes fresh stats.
        """
        await self._session_logger_getter().save_interaction(
            self._messages,
            self._stats_getter(),
            self._config_getter(),
            self._tool_manager_getter(),
            None,
        )
        # Prepare and commit the new root binding before clearing live history.
        await self._reset_session(False)
        self._messages.reset(self._messages[:1])

        self._publish_fresh_stats()
        self.apply_active_model_pricing()

        self._middleware_pipeline_getter().reset()
        self._tool_manager_getter().reset_all()
        # A history clear starts a fresh context: the working set is
        # session-scoped state and does not survive it (the dispatcher-level
        # revision registry does — it is tree truth, not session state).
        self._file_state_context.reset()
