"""Context-injection collaborator for AgentLoop (ALL cluster: steering/skills).

Owns the skill-loaded detection, the synthetic skill expansion, and the
mentioned-file expansion, all extracted from _loop.py. The synthetic skill
expansion keeps its separate direct path: it builds the skill result itself
and records the call/result messages without going through single-tool
execution. Mentioned-file expansion instead drives read_file through the
single-tool execution callback, with the raw ResolvedToolCall arguments.

Also owns the user-context construction with its callback bracketing
(before_commit/on_commit around the append, on_injected_event around each
expansion event, on_context_ready completing-or-unwinding after expansion),
the pending injected-message buffer, the tracked steering-expansion tasks,
and their settlement (drain_pending appends pending messages to the
transcript; settle joins or cancels the tracked tasks). The injector borrows
the transcript recorder and the authoritative MessageList; live loop values
(skill manager, tool manager, harness files, cwd, current model context)
are read through getters, so manager replacement and relocation stay
visible. The patched module-level ``build_skill_result`` surface is resolved
at invocation time through a facade callback, so test patches of the _loop
global keep intercepting, and persistence goes through the late-bound
``save_messages`` callback so instance patches keep intercepting.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
import json
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from pydantic import JsonValue

from chartreux.core.agent_loop._transcript_integrity import check_tool_message_admission
from chartreux.core.agent_loop._transcript_recorder import TranscriptRecorder
from chartreux.core.agent_loop.errors import AgentLoopError
from chartreux.core.autocompletion.path_prompt import build_path_prompt_payload
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.events import (
    BaseEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
    UserMessageEvent,
)
from chartreux.core.hooks.models import HookEvent
from chartreux.core.llm.format import ResolvedToolCall
from chartreux.core.llm_models import (
    CONTEXT_INJECTION_TOOL_PAIR_V1,
    FunctionCall,
    ImageAttachment,
    LLMMessage,
    ManualShellContext,
    PersistedToolResult,
    Role,
    ToolCall,
    posting_time,
)
from chartreux.core.message_list import MessageList
from chartreux.core.skills.manager import SkillManager
from chartreux.core.timing import InvocationTiming
from chartreux.core.tools.builtins.read_file import ReadFileArgs
from chartreux.core.tools.builtins.skill import (
    Skill as SkillTool,
    SkillArgs,
    SkillResult,
    skill_content_marker,
)
from chartreux.core.tools.manager import NoSuchToolError, ToolManager
from chartreux.core.tools.ui import ToolUIDataAdapter
from chartreux.user_content import UserResource


class ContextInjector:
    """Own skill/mention expansion, user-context construction, and steering state.

    ``skill_already_loaded`` scans the current model context for a skill's
    content marker. ``inject_skill`` expands an invoked ``/skill`` command
    on its separate direct path (no single-tool execution). ``inject_mentions``
    expands ``@mentioned`` files by executing read_file through the loop's
    single-tool path, outside any batch window. ``inject_user_context``
    brackets the user-context construction with the commit/injection/ready
    callbacks and tracks the calling task while expansion runs;
    ``drain_pending`` appends queued injected messages to the transcript;
    ``settle`` joins (or cancels) the tracked steering tasks.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        messages: MessageList,
        transcript_recorder: TranscriptRecorder,
        skill_manager_getter: Callable[[], SkillManager],
        tool_manager_getter: Callable[[], ToolManager],
        harness_files_getter: Callable[[], HarnessFilesManager],
        cwd_getter: Callable[[], Path],
        model_context_getter: Callable[[], list[LLMMessage]],
        record_tool_call: Callable[[ToolCall], ToolCall],
        record_tool_call_event: Callable[[ToolCallEvent], ToolCallEvent],
        execute_tool: Callable[
            [ResolvedToolCall],
            AsyncGenerator[ToolResultEvent | ToolStreamEvent | HookEvent],
        ],
        build_skill_result: Callable[..., Awaitable[SkillResult]],
        save_messages: Callable[[], Awaitable[None]],
    ) -> None:
        # Borrowed shared ownership: the authoritative MessageList and the
        # transcript recorder are the loop's own objects, never cloned or
        # replaced.
        self._messages = messages
        self._transcript_recorder = transcript_recorder
        # Live getters: the skill and tool managers are replaced by
        # configuration transactions; harness files and cwd are reassigned
        # on relocation; the model context is read at detection time.
        self._skill_manager_getter = skill_manager_getter
        self._tool_manager_getter = tool_manager_getter
        self._harness_files_getter = harness_files_getter
        self._cwd_getter = cwd_getter
        self._model_context_getter = model_context_getter
        # Narrow late-bound callbacks: call/result recording resolves through
        # the loop at call time, so instance patches of the loop's recording
        # methods keep intercepting.
        self._record_tool_call = record_tool_call
        self._record_tool_call_event = record_tool_call_event
        # Mentioned-file expansion executes through the loop's single-tool
        # path (its compatibility delegate), so execution patches keep
        # intercepting.
        self._execute_tool = execute_tool
        # Facade callback: the patched module-level build_skill_result
        # surface resolves through the _loop global at invocation time.
        self._build_skill_result = build_skill_result
        # Late-bound persistence: instance patches of the loop's save method
        # keep intercepting the post-construction save.
        self._save_messages = save_messages
        # Owned steering state: the pending injected-message buffer (session
        # reset replaces it through the compatibility setter) and the tracked
        # steering-expansion tasks.
        self._pending_injected_messages: list[LLMMessage] = []
        self._steering_injections: set[asyncio.Task[Any]] = set()

    @property
    def pending_injected_messages(self) -> list[LLMMessage]:
        """Compatibility view of the pending injected-message buffer."""
        return self._pending_injected_messages

    @pending_injected_messages.setter
    def pending_injected_messages(self, messages: list[LLMMessage]) -> None:
        self._pending_injected_messages = messages

    @property
    def steering_injections(self) -> set[asyncio.Task[Any]]:
        """Read-only compatibility view of the tracked steering tasks."""
        return self._steering_injections

    async def inject_user_context(  # noqa: PLR0913
        self,
        content: str,
        *,
        as_message: bool = False,
        inject_implicit: bool = False,
        images: list[ImageAttachment] | None = None,
        input_text: str | None = None,
        resources: list[UserResource] | None = None,
        client_message_id: str | None = None,
        manual_shell: ManualShellContext | None = None,
        before_commit: Callable[[], None] | None = None,
        on_commit: Callable[[UserMessageEvent], None] | None = None,
        on_injected_event: Callable[[BaseEvent], None] | None = None,
        on_context_ready: Callable[[], object] | None = None,
    ) -> list[BaseEvent]:
        task = asyncio.current_task()
        if on_context_ready is not None and task is not None:
            self._steering_injections.add(task)
        try:
            return await self._inject_user_context(
                content,
                as_message=as_message,
                inject_implicit=inject_implicit,
                images=images,
                input_text=input_text,
                resources=resources,
                client_message_id=client_message_id,
                manual_shell=manual_shell,
                before_commit=before_commit,
                on_commit=on_commit,
                on_injected_event=on_injected_event,
                on_context_ready=on_context_ready,
            )
        finally:
            if task is not None:
                self._steering_injections.discard(task)

    async def _inject_user_context(  # noqa: PLR0913
        self,
        content: str,
        *,
        as_message: bool,
        inject_implicit: bool,
        images: list[ImageAttachment] | None,
        input_text: str | None,
        resources: list[UserResource] | None,
        client_message_id: str | None,
        manual_shell: ManualShellContext | None,
        before_commit: Callable[[], None] | None,
        on_commit: Callable[[UserMessageEvent], None] | None,
        on_injected_event: Callable[[BaseEvent], None] | None,
        on_context_ready: Callable[[], object] | None,
    ) -> list[BaseEvent]:
        events: list[BaseEvent] = []
        if as_message:
            message = LLMMessage(
                role=Role.user,
                content=content,
                message_id=client_message_id or str(uuid4()),
                posted_at=posting_time(),
                images=images or None,
                input_text=input_text,
                resources=resources or None,
                manual_shell=manual_shell,
            )
            if message.message_id is None:
                raise AgentLoopError("User message must have a message_id")
            event = UserMessageEvent(
                content=input_text if input_text is not None else content,
                message_id=message.message_id,
                posted_at=message.posted_at,
                images=list(message.images or []),
                resources=list(message.resources or []),
            )
            if before_commit is not None:
                before_commit()
            self._messages.append(message)
            if on_commit is not None:
                on_commit(event)
            events.append(event)
            try:
                if inject_implicit:
                    async for event in self.inject_skill(content):
                        events.append(event)
                        if on_injected_event is not None:
                            on_injected_event(event)
                    async for event in self.inject_mentions(content):
                        events.append(event)
                        if on_injected_event is not None:
                            on_injected_event(event)
            finally:
                # Only release the parent's waits after expansion has completed
                # or unwound. The turn also joins this task before continuing.
                if on_context_ready is not None:
                    on_context_ready()
        else:
            self._messages.append(
                LLMMessage(
                    role=Role.user,
                    content=content,
                    injected=True,
                    images=images or None,
                    input_text=input_text,
                    resources=resources or None,
                    manual_shell=manual_shell,
                )
            )
        await self._save_messages()
        return events

    def drain_pending(self) -> bool:
        if not self._pending_injected_messages:
            return False
        for injected in self._pending_injected_messages:
            self._messages.append(injected)
        self._pending_injected_messages.clear()
        return True

    async def settle(self, *, cancel: bool = False) -> None:
        while tasks := self._steering_injections.copy():
            if cancel:
                for task in tasks:
                    task.cancel()
            try:
                await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
            except asyncio.CancelledError:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

    def skill_already_loaded(self, name: str) -> bool:
        marker = skill_content_marker(name)
        return any(
            m.role == Role.tool and m.name == "skill" and marker in (m.content or "")
            for m in self._model_context_getter()
        )

    async def inject_skill(self, user_msg: str) -> AsyncGenerator[BaseEvent, None]:
        parsed = self._skill_manager_getter().parse_skill_command(user_msg)
        if parsed is None:
            return
        skill_info = self._skill_manager_getter().get_skill(parsed.name)
        if skill_info is None:
            return

        timing = InvocationTiming()
        timing.start()
        result = await self._build_skill_result(
            skill_info, already_loaded=self.skill_already_loaded(parsed.name)
        )
        duration = timing.finish()
        call_id = str(uuid4())
        tool_class = self._tool_manager_getter().available_tools.get("skill", SkillTool)
        call_event = ToolCallEvent(
            tool_call_id=call_id,
            tool_call_index=0,
            tool_name="skill",
            tool_class=tool_class,
            args=SkillArgs(name=parsed.name),
        )
        call_event = call_event.model_copy(
            update={
                "presentation": ToolUIDataAdapter(
                    tool_class, harness_files=self._harness_files_getter()
                ).get_call_presentation(call_event)
            }
        )
        result_event = ToolResultEvent(
            tool_name="skill",
            tool_class=tool_class,
            result=result,
            duration=duration,
            tool_call_id=call_id,
        )
        result_event = result_event.model_copy(
            update={
                "presentation": ToolUIDataAdapter(
                    tool_class, harness_files=self._harness_files_getter()
                ).get_result_presentation(result_event)
            }
        )

        call_event = self._record_tool_call_event(call_event)
        skill_assistant = LLMMessage(
            role=Role.assistant,
            content="",
            tool_calls=[
                self._record_tool_call(
                    ToolCall(
                        id=call_id,
                        index=0,
                        function=FunctionCall(
                            name="skill", arguments=json.dumps({"name": parsed.name})
                        ),
                        presentation=call_event.presentation,
                    )
                )
            ],
        )
        # Round-5 Y: the injector OWNS this assistant message — stamp the
        # private origin marker immediately before its initial append (never
        # derived from call ids, content, or any provider-supplied value).
        # The marker means "originated in ContextInjector", not "settled";
        # the pair's result below is unmarked (its ownership and settlement
        # stay established by the existing integrity checks).
        skill_assistant._context_injection_origin = CONTEXT_INJECTION_TOOL_PAIR_V1
        self._messages.append(skill_assistant)
        result_text = "\n".join(f"{k}: {v}" for k, v in result.model_dump().items())
        skill_result_message = LLMMessage(
            role=Role.tool,
            tool_call_id=call_id,
            name="skill",
            content=result_text,
            tool_result=PersistedToolResult(
                output=cast(dict[str, JsonValue], result.model_dump(mode="json")),
                duration=duration,
                presentation=result_event.presentation,
            ),
        )
        # PI1/T (S1 append boundary): the direct skill-expansion append is an
        # insert path like any other — it may not create a duplicate tool
        # result or an unrepairable [user, tool] adjacency (fail closed with
        # the typed local diagnostic).
        check_tool_message_admission(
            self._messages, len(self._messages), skill_result_message
        )
        self._messages.append(skill_result_message)

        yield call_event
        yield result_event

    async def inject_mentions(self, user_msg: str) -> AsyncGenerator[BaseEvent, None]:
        payload = build_path_prompt_payload(user_msg, base_dir=self._cwd_getter())
        file_resources = [r for r in payload.resources if r.kind == "file"]
        if not file_resources:
            return
        try:
            tool_instance = self._tool_manager_getter().get("read_file")
        except NoSuchToolError:
            return
        tool_class = type(tool_instance)

        for resource in file_resources:
            file_path = str(resource.path)
            call_id = str(uuid4())
            mention_assistant = LLMMessage(
                role=Role.assistant,
                content="",
                tool_calls=[
                    self._record_tool_call(
                        ToolCall(
                            id=call_id,
                            index=0,
                            function=FunctionCall(
                                name="read_file",
                                arguments=json.dumps({"file_path": file_path}),
                            ),
                        )
                    )
                ],
            )
            # Round-5 Y: the injector OWNS this assistant message — stamp the
            # private origin marker immediately before its initial append.
            mention_assistant._context_injection_origin = CONTEXT_INJECTION_TOOL_PAIR_V1
            self._messages.append(mention_assistant)
            tool_call = ResolvedToolCall(
                tool_name="read_file",
                tool_class=tool_class,
                validated_args=ReadFileArgs(file_path=file_path),
                call_id=call_id,
            )
            call_event = ToolCallEvent(
                tool_call_id=call_id,
                tool_call_index=0,
                tool_name="read_file",
                tool_class=tool_class,
                args=ReadFileArgs(file_path=file_path),
            )
            call_event = call_event.model_copy(
                update={
                    "presentation": ToolUIDataAdapter(
                        tool_class, harness_files=self._harness_files_getter()
                    ).get_call_presentation(call_event)
                }
            )
            call_event = self._record_tool_call_event(call_event)
            self._transcript_recorder.record_presentation(call_event)
            yield call_event
            async for event in self._execute_tool(tool_call):
                yield event
