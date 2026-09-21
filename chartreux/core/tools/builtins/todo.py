from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from enum import StrEnum, auto
import json

from pydantic import BaseModel, Field, computed_field

from chartreux.core.events import ToolResultEvent, ToolStreamEvent
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from chartreux.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from chartreux.utils.tool_presentation import ToolEffectKind


class TodoStatus(StrEnum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    CANCELLED = auto()


class TodoPriority(StrEnum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


class TodoItem(BaseModel):
    id: str = Field(
        description="Stable unique identifier for the task, reused across updates"
    )
    content: str = Field(description="Brief description of the task")
    status: TodoStatus = Field(
        default=TodoStatus.PENDING,
        description="Current status of the task: pending, in_progress, completed, cancelled",
    )
    priority: TodoPriority = Field(
        default=TodoPriority.MEDIUM,
        description="Priority level of the task: high, medium, low",
    )


class TodoArgs(BaseModel):
    action: str = Field(
        description="Required on every call: 'read' to view the current list, or 'write' to replace it"
    )
    todos: list[TodoItem] | None = Field(
        default=None,
        description="Required when action='write': the full todo list, which replaces the previous one",
    )


class TodoResult(BaseModel):
    verb: str
    todos: list[TodoItem]
    total_count: int

    @computed_field
    @property
    def message(self) -> str:
        return f"{self.verb} {self.total_count} todos"


class TodoConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_todos: int = 100


class TodoState(BaseToolState):
    todos: list[TodoItem] = Field(default_factory=list)

    @classmethod
    def replay(cls, messages: Sequence[LLMMessage]) -> TodoState:
        """Restore the last successful todo write recorded in a transcript."""
        todos: list[TodoItem] = []
        calls: dict[str, list[TodoItem]] = {}
        for message in messages:
            if message.context_boundary == "compaction":
                boundary_state = (
                    message.tool_result.output if message.tool_result else {}
                ).get("todo_state")
                if isinstance(boundary_state, dict):
                    boundary_state = boundary_state.get("todos")
                if isinstance(boundary_state, list):
                    try:
                        todos = [
                            TodoItem.model_validate(todo) for todo in boundary_state
                        ]
                    except (TypeError, ValueError):
                        pass
                continue
            if message.role is Role.assistant:
                for call in message.tool_calls or []:
                    if call.id is None or call.function.name != "todo":
                        continue
                    try:
                        args = TodoArgs.model_validate_json(
                            call.function.arguments or "{}"
                        )
                    except (ValueError, json.JSONDecodeError):
                        continue
                    if args.action == "write" and args.todos is not None:
                        calls[call.id] = args.todos
                continue
            if (
                message.role is Role.tool
                and message.tool_call_id in calls
                and message.tool_result is not None
                and not message.tool_result.cancelled
            ):
                todos = calls[message.tool_call_id]
        return cls(todos=todos)


class Todo(
    BaseTool[TodoArgs, TodoResult, TodoConfig, TodoState],
    ToolUIData[TodoArgs, TodoResult],
):
    effect_kind = ToolEffectKind.TODO

    @classmethod
    def format_call_display(cls, args: TodoArgs) -> ToolCallDisplay:
        match args.action:
            case "read":
                return ToolCallDisplay(
                    summary="Reading todos",
                    verb="Retrieving",
                    message="todos",
                    settled_verb="Retrieved",
                    settled_message="todos",
                )
            case "write":
                count = len(args.todos) if args.todos else 0
                return ToolCallDisplay(
                    summary=f"Writing {count} todos",
                    verb="Updating",
                    message=f"{count} todos",
                    settled_verb="Updated",
                    settled_message=f"{count} todos",
                )
            case _:
                return ToolCallDisplay(
                    summary=f"Unknown action: {args.action}",
                    verb="Running",
                    message=f"unknown todo action: {args.action}",
                    settled_verb="Ran",
                    settled_message=f"unknown todo action: {args.action}",
                )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, TodoResult):
            return ToolResultDisplay(success=True, message="Success")

        result = event.result
        return ToolResultDisplay(
            success=True, verb=result.verb, message=f"{result.total_count} todos"
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Managing todos"

    async def run(
        self, args: TodoArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | TodoResult, None]:
        match args.action:
            case "read":
                yield self._read_todos()
            case "write":
                yield self._write_todos(args.todos or [])
            case _:
                raise ToolError(
                    f"Invalid action '{args.action}'. Use 'read' or 'write'."
                )

    def _read_todos(self) -> TodoResult:
        return TodoResult(
            verb="Retrieved", todos=self.state.todos, total_count=len(self.state.todos)
        )

    def _write_todos(self, todos: list[TodoItem]) -> TodoResult:
        if len(todos) > self.config.max_todos:
            raise ToolError(f"Cannot store more than {self.config.max_todos} todos")

        ids = [todo.id for todo in todos]
        if len(ids) != len(set(ids)):
            raise ToolError("Todo IDs must be unique")

        self.state.todos = todos

        return TodoResult(
            verb="Updated", todos=self.state.todos, total_count=len(self.state.todos)
        )
