"""Single-tool execution collaborator for AgentLoop (ALL cluster: executor).

Owns one complete tool lifecycle. ``invoke_tool`` is the internal
invocation helper retaining today's snapshot/admission/invoke span, and
``process`` runs the whole per-call envelope around it (lookup,
serialization, pre-tool pipeline, decision, cancellation/failure
finalization). The executor borrows the dangerous-command grant store
and the checkpoint recorder, and reads live loop values (stats, tool
manager, harness files) through getters, so session reset, manager
replacement, and relocation stay visible. The invocation context is
built by the loop through the synchronous ``build_invoke_context``
callback, so no await or yielding helper sits between the execution
admission gate and the actual invocation. The hook operations
(serialize, pre-tool pipeline, skip, post-tool finalization, cancel
finalization), the decision and wait-registration ports, and the
transcript-response and file-observation callbacks are explicit
late-bound callbacks: the hook mixin stays a mixin on the loop, and
instance patches of the loop's methods keep intercepting.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from chartreux.core.agent_loop._dangerous_command_grants import (
    DangerousCommandGrantStore,
)
from chartreux.core.agent_loop._execution_admission import (
    revalidate_execution_admission,
)
from chartreux.core.agent_loop._loop_types import ToolDecision, ToolExecutionResponse
from chartreux.core.agent_loop_hooks import PostToolFinalization, _PreToolResolution
from chartreux.core.checkpoints import CheckpointRecorder
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.events import (
    ToolCancellationOrigin,
    ToolResultEvent,
    ToolStreamEvent,
)
from chartreux.core.hooks.models import HookEvent
from chartreux.core.llm.format import ResolvedToolCall
from chartreux.core.session_types import AgentStats
from chartreux.core.timing import InvocationTiming
from chartreux.core.tools.base import (
    BaseTool,
    CancellableToolResult,
    InvokeContext,
    ToolError,
    ToolPermission,
    ToolPermissionError,
)
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.ui import ToolUIDataAdapter
from chartreux.core.utils import (
    TOOL_ERROR_TAG,
    CancellationReason,
    get_user_cancellation_message,
)
from chartreux.observability.logging import logger
from chartreux.utils import VIBE_WARNING_TAG


class ToolExecutor:
    """Own one complete tool lifecycle for a single resolved tool call.

    No persistent invocation state: each execution owns its rewritten
    call/input, decision, timing, result, and the existing tool_started
    flag semantics (set before the snapshot, read by the cancel path).
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        stats_getter: Callable[[], AgentStats],
        tool_manager_getter: Callable[[], ToolManager],
        dangerous_command_grants: DangerousCommandGrantStore,
        checkpoint_recorder: CheckpointRecorder,
        harness_files_getter: Callable[[], HarnessFilesManager],
        build_invoke_context: Callable[[ResolvedToolCall], InvokeContext],
        handle_tool_skip: Callable[
            [ResolvedToolCall, ToolDecision],
            AsyncGenerator[ToolResultEvent | HookEvent],
        ],
        run_post_tool_and_finalize: Callable[
            [ResolvedToolCall, PostToolFinalization], AsyncGenerator[HookEvent]
        ],
        observe_file_state: Callable[[str, BaseModel], None],
        serialize_tool_input: Callable[[ResolvedToolCall], dict[str, Any]],
        run_pre_tool_pipeline: Callable[
            [ResolvedToolCall, dict[str, Any]],
            Awaitable[tuple[list[HookEvent], _PreToolResolution]],
        ],
        should_execute_tool: Callable[
            [BaseTool, BaseModel, str | None], Awaitable[ToolDecision]
        ],
        is_steering_cancellation: Callable[[str], bool],
        finalize_cancelled_tool: Callable[..., AsyncGenerator[HookEvent]],
        tool_failure_event: Callable[[ResolvedToolCall, str], ToolResultEvent],
        handle_tool_response: Callable[[ResolvedToolCall, str], None],
    ) -> None:
        # Borrowed shared ownership: the grant store and the checkpoint
        # recorder are the loop's own objects, never cloned or replaced.
        self._dangerous_command_grants = dangerous_command_grants
        self._checkpoint_recorder = checkpoint_recorder
        # Live getters: stats is replaced on session reset, the tool manager
        # is replaced by configuration transactions, and harness files are
        # replaced on relocation; all are resolved at use time, never
        # captured at construction.
        self._stats_getter = stats_getter
        self._tool_manager_getter = tool_manager_getter
        self._harness_files_getter = harness_files_getter
        # Synchronous facade callback: the loop builds the invocation
        # context, keeping every live turn value on the loop's side of the
        # seam; no await or yielding helper sits between the admission gate
        # and the invocation.
        self._build_invoke_context = build_invoke_context
        # Explicit hook operations and the file-observation callback:
        # late-bound through the loop, so instance patches of the mixin
        # methods keep intercepting.
        self._handle_tool_skip = handle_tool_skip
        self._run_post_tool_and_finalize = run_post_tool_and_finalize
        self._observe_file_state = observe_file_state
        # The remaining hook operations, the decision and wait-registration
        # ports, and the transcript-response callbacks: likewise late-bound
        # through the loop, so instance patches keep intercepting (patch
        # riders: _serialize_tool_input at test_agent_tool_call.py:1265,
        # _should_execute_tool and _is_steering_cancellation in
        # test_agent_wait_steering.py:395-402/:454-461/:499).
        self._serialize_tool_input = serialize_tool_input
        self._run_pre_tool_pipeline = run_pre_tool_pipeline
        self._should_execute_tool = should_execute_tool
        self._is_steering_cancellation = is_steering_cancellation
        self._finalize_cancelled_tool = finalize_cancelled_tool
        self._tool_failure_event = tool_failure_event
        self._handle_tool_response = handle_tool_response

    async def process(
        self, tool_call: ResolvedToolCall
    ) -> AsyncGenerator[ToolResultEvent | ToolStreamEvent | HookEvent]:
        # Run one complete tool lifecycle: lookup, serialization, pre-tool
        # pipeline, decision, invocation, and the cancellation/failure
        # envelope. The envelope ordering is the pre-move ordering exactly:
        # tool_started is set before the snapshot inside invoke_tool; the
        # pre-hook pipeline lies outside the decision/invocation catch; the
        # cancelled post-hook collection stays shielded in the mixin; and
        # observation then success accounting stay after post-tool
        # finalization.
        try:
            tool_instance = self._tool_manager_getter().get(tool_call.tool_name)
        except Exception as exc:
            error_msg = f"Error getting tool '{tool_call.tool_name}': {exc}"
            yield self._tool_failure_event(tool_call, error_msg)
            return

        try:
            tool_input = self._serialize_tool_input(tool_call)
        except Exception as exc:
            error_msg = (
                f"<{TOOL_ERROR_TAG}>Failed to serialize tool input for "
                f"'{tool_call.tool_name}': {exc}</{TOOL_ERROR_TAG}>"
            )
            self._stats_getter().tool_calls_failed += 1
            yield ToolResultEvent(
                tool_name=tool_call.tool_name,
                tool_class=tool_call.tool_class,
                error=error_msg,
                tool_call_id=tool_call.call_id,
            )
            self._handle_tool_response(tool_call, error_msg)
            return

        events, resolution = await self._run_pre_tool_pipeline(tool_call, tool_input)
        for ev in events:
            yield ev
        if resolution.denial_event is not None:
            yield resolution.denial_event
            return
        tool_call = resolution.tool_call
        tool_input = resolution.tool_input

        decision: ToolDecision | None = None
        tool_started = False
        timing = InvocationTiming()
        try:
            decision = await self._should_execute_tool(
                tool_instance, tool_call.validated_args, tool_call.call_id
            )

            if decision.verdict == ToolExecutionResponse.SKIP:
                async for ev in self._handle_tool_skip(tool_call, decision):
                    yield ev
                return

            tool_started = True
            async for ev in self.invoke_tool(
                tool_call, tool_instance, tool_input, decision, timing
            ):
                yield ev

        except asyncio.CancelledError as exc:
            steering = self._is_steering_cancellation(tool_call.call_id)
            timing.finish()
            cancel = str(
                get_user_cancellation_message(CancellationReason.TOOL_INTERRUPTED)
            )
            logger.info(
                "Tool call cancelled tool=%s tool_call_id=%s outcome=cancelled",
                tool_call.tool_name,
                tool_call.call_id,
            )
            self._stats_getter().tool_calls_failed += 1
            yield ToolResultEvent(
                tool_name=tool_call.tool_name,
                tool_class=tool_call.tool_class,
                error=cancel,
                cancelled=True,
                cancellation_origin=(
                    ToolCancellationOrigin.STEERING if steering else None
                ),
                duration=timing.duration,
                error_write_paths=getattr(exc, "write_paths", None) or [],
                error_write_paths_unknown=bool(
                    getattr(exc, "write_paths_unknown", False)
                ),
                error_mutation_roots=getattr(exc, "mutation_roots", None) or [],
                error_mutation_roots_unknown=bool(
                    getattr(exc, "mutation_roots_unknown", False)
                ),
                tool_call_id=tool_call.call_id,
            )
            async for ev in self._finalize_cancelled_tool(
                tool_call,
                tool_input,
                cancel,
                tool_started=tool_started,
                duration=timing.duration,
            ):
                yield ev
            raise

        except Exception as exc:
            timing.finish()
            # One prefix for both: the model reads `error`, the client `display`.
            failure = f"{tool_instance.get_name()} failed: "
            error_msg = f"<{TOOL_ERROR_TAG}>{failure}{exc}</{TOOL_ERROR_TAG}>"
            if isinstance(exc, ToolPermissionError):
                logger.info(
                    "Tool call denied tool=%s tool_call_id=%s outcome=denied",
                    tool_call.tool_name,
                    tool_call.call_id,
                )
                self._stats_getter().tool_calls_agreed -= 1
                self._stats_getter().tool_calls_rejected += 1
            else:
                logger.warning(
                    "Tool call failed tool=%s tool_call_id=%s outcome=error error=%s",
                    tool_call.tool_name,
                    tool_call.call_id,
                    type(exc).__name__,
                )
                self._stats_getter().tool_calls_failed += 1
            yield ToolResultEvent(
                tool_name=tool_call.tool_name,
                tool_class=tool_call.tool_class,
                error=error_msg,
                duration=timing.duration,
                error_display=(
                    f"{failure}{exc.display}" if isinstance(exc, ToolError) else None
                ),
                # R7: a failed or timed-out shell run may still have written
                # files; the tool attaches its write footprint to the raised
                # error so the submission scan keeps covering it. A1: the
                # mutation-roots footprint rides the error the same way.
                error_write_paths=getattr(exc, "write_paths", None) or [],
                error_write_paths_unknown=bool(
                    getattr(exc, "write_paths_unknown", False)
                ),
                error_mutation_roots=getattr(exc, "mutation_roots", None) or [],
                error_mutation_roots_unknown=bool(
                    getattr(exc, "mutation_roots_unknown", False)
                ),
                tool_call_id=tool_call.call_id,
            )
            async for ev in self._run_post_tool_and_finalize(
                tool_call,
                PostToolFinalization(
                    tool_input=tool_input,
                    tool_status="failure",
                    response_status="failure",
                    tool_error=str(exc),
                    duration_ms=timing.duration_ms,
                    duration=timing.duration,
                    initial_text=error_msg,
                ),
            ):
                yield ev

    async def invoke_tool(
        self,
        tool_call: ResolvedToolCall,
        tool_instance: BaseTool,
        tool_input: dict[str, Any],
        decision: ToolDecision,
        timing: InvocationTiming,
    ) -> AsyncGenerator[ToolResultEvent | ToolStreamEvent | HookEvent]:
        self._stats_getter().tool_calls_agreed += 1

        snapshot = await asyncio.to_thread(
            tool_instance.get_file_snapshot, tool_call.validated_args
        )
        if snapshot is not None:
            self._checkpoint_recorder.add_snapshot(snapshot)

        # The snapshot await is a world-change window: dangerous-command
        # authority — or any other permission dimension — may have been
        # revoked while it ran. Re-run the full resolution at the execution
        # handoff, after the snapshot and immediately before invoke, so no
        # unguarded await remains between the last admission decision and the
        # actual execution. The gate covers every execution path of the tool,
        # including the shell tool's client-terminal delegation.
        admission_denial = revalidate_execution_admission(
            tool_instance,
            tool_call.validated_args,
            self._tool_manager_getter(),
            self._dangerous_command_grants,
            decision.admission,
        )
        if admission_denial is not None:
            self._stats_getter().tool_calls_agreed -= 1
            logger.info(
                "Tool call denied at execution admission tool=%s tool_call_id=%s "
                "outcome=denied",
                tool_call.tool_name,
                tool_call.call_id,
            )
            async for ev in self._handle_tool_skip(
                tool_call,
                ToolDecision(
                    verdict=ToolExecutionResponse.SKIP,
                    approval_type=ToolPermission.NEVER,
                    feedback=admission_denial,
                ),
            ):
                yield ev
            return

        timing.start()
        logger.debug(
            "Tool call starting tool=%s tool_call_id=%s",
            tool_call.tool_name,
            tool_call.call_id,
        )
        result_model = None
        ctx = self._build_invoke_context(tool_call)
        async for item in tool_instance.invoke(ctx=ctx, **tool_call.args_dict):
            if isinstance(item, ToolStreamEvent):
                yield item
            else:
                result_model = item

        duration = timing.finish()
        assert duration is not None
        if result_model is None:
            raise ToolError("Tool did not yield a result")

        result_dict = result_model.model_dump(mode="json")
        text = "\n".join(f"{k}: {v}" for k, v in result_dict.items())
        extra = tool_instance.get_result_extra(result_model)
        if extra:
            text += "\n\n" + extra
        if decision.result_note:
            text += (
                f"\n\n<{VIBE_WARNING_TAG}>{decision.result_note}</{VIBE_WARNING_TAG}>"
            )

        result_cancelled = (
            isinstance(result_model, CancellableToolResult) and result_model.cancelled
        )
        images = (
            None if result_cancelled else tool_instance.get_result_images(result_model)
        )
        result_event = ToolResultEvent(
            tool_name=tool_call.tool_name,
            tool_class=tool_call.tool_class,
            result=result_model,
            cancelled=result_cancelled,
            duration=duration,
            tool_call_id=tool_call.call_id,
        )
        result_event = result_event.model_copy(
            update={
                "presentation": ToolUIDataAdapter(
                    tool_call.tool_class, harness_files=self._harness_files_getter()
                ).get_result_presentation(result_event)
            }
        )
        yield result_event
        async for ev in self._run_post_tool_and_finalize(
            tool_call,
            PostToolFinalization(
                tool_input=tool_input,
                tool_status="cancelled" if result_cancelled else "success",
                response_status="success",
                tool_output=result_dict,
                tool_presentation=result_event.presentation,
                images=images,
                duration_ms=duration * 1000.0,
                duration=duration,
                initial_text=text,
            ),
        ):
            yield ev
        if not result_cancelled:
            # Observation intake: a successful authorized file interaction
            # (read / edit / write) enrolls the path and publishes its revision
            # to the dispatcher-level registry at observation time.
            self._observe_file_state(tool_call.tool_name, result_model)
        self._stats_getter().tool_calls_succeeded += 1
        logger.info(
            "Tool call completed tool=%s tool_call_id=%s duration_ms=%d outcome=%s",
            tool_call.tool_name,
            tool_call.call_id,
            int(duration * 1000),
            "cancelled" if result_cancelled else "success",
        )
