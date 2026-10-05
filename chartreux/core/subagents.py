from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
import enum
import json
import re
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chartreux.core.events import (
    AssistantEvent,
    BaseEvent,
    ToolResultEvent,
    ToolStreamEvent,
)
from chartreux.core.launch_types import LaunchConfig, LaunchToolOverride
from chartreux.core.tools.ui import ToolUIDataAdapter

__all__ = ["LaunchConfig", "LaunchToolOverride"]

if TYPE_CHECKING:
    from chartreux.core.tools.base import InvokeContext


def normalize_task_summary(
    value: str | None, *, fallback: str | None = None
) -> str | None:
    """Normalize a supplied task summary or its fallback preview."""
    if value is None:
        value = fallback
    if value is None:
        return None
    return " ".join(value.split())[:240]


class TaskArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(description="The task for the agent to perform")
    task_summary: str | None = Field(
        default=None,
        description="Optional short summary of the task for retained-agent status.",
    )
    agent_type: str = Field(
        default="worker",
        description=(
            "Agent type: profile name for a new agent instance (default worker). "
            "To continue a retained instance, use agent_id instead, with background: true."
        ),
    )
    background: bool = Field(
        default=True,
        description=(
            "If true, launch the subagent in the background and return a handle "
            "instead of blocking until completion. Set false only when you need "
            "the result immediately."
        ),
    )
    agent_id: str | None = Field(
        default=None,
        description=(
            "Retained agent instance handle to continue with background: true. "
            "If omitted, create a new instance using the agent_type profile name."
        ),
    )
    replace_run: bool = Field(
        default=False,
        description="Stop busy background work before reusing agent_id in the same conversation.",
    )
    config: LaunchConfig | None = Field(
        default=None,
        description="Optional semantic launch configuration overrides for this child.",
    )

    @model_validator(mode="before")
    @classmethod
    def _omit_null_config(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        fan_out = value.get("fan_out", False)
        if fan_out is not False and fan_out is not None:
            raise ValueError(
                "Roles are single presets. Launch separate tasks with explicit "
                "presets/models for multiple agents."
            )
        value = {key: item for key, item in value.items() if key != "fan_out"}

        config = value.get("config")
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except json.JSONDecodeError as error:
                raise ValueError("config must be a valid JSON object") from error
            if config is not None and not isinstance(config, dict):
                raise ValueError("config JSON must decode to an object")

        if config is None:
            return {key: item for key, item in value.items() if key != "config"}
        return {**value, "config": config}

    @model_validator(mode="after")
    def _reject_instance_handle_as_type(self) -> TaskArgs:
        if re.fullmatch(r"agent-\d+", self.agent_type):
            raise ValueError(
                "agent_type takes a profile name; to continue an existing agent "
                "instance, pass its handle as agent_id (with background: true)"
            )
        return self

    @field_validator("task_summary")
    @classmethod
    def _normalize_task_summary(cls, value: str | None) -> str | None:
        return normalize_task_summary(value)


class LaunchOutcome(enum.StrEnum):
    LAUNCHED = "launched"
    ALREADY_STOPPING = "already_stopping"
    ALREADY_FINISHING = "already_finishing"
    REJECTED_RESERVATION = "rejected_reservation"


class TaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str = Field(description="The accumulated response from the subagent")
    turns_used: int = Field(description="Number of turns the subagent used")
    completed: bool = Field(description="Whether the task completed normally")
    launch_outcome: LaunchOutcome | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    status: Literal["launched"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Launch state for a background launch acknowledgment; None for task results.",
    )
    agent_id: str | None = Field(
        default=None,
        description="Stable agent handle. Populated for background launches; None for foreground.",
    )
    run_id: str | None = Field(
        default=None,
        description="Unique run identifier for this invocation. Populated for background launches; None for foreground.",
    )
    stop_reason: RunStopReason | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additive completion metadata, including provider failover visibility.",
    )


class RunStopReason(enum.StrEnum):
    """Structured run outcome; values match the public app-server contract."""

    USER_CANCELLED = "user_cancelled"
    ORCHESTRATOR_CANCELLED = "orchestrator_cancelled"
    RETASKED = "retasked"
    BUDGET_EXCEEDED = "budget_exceeded"
    BUDGET_UNVERIFIABLE = "budget_unverifiable"
    ERROR = "error"


class CancelOutcome(enum.StrEnum):
    STOP_REQUESTED = "stop_requested"
    ALREADY_STOPPING = "already_stopping"
    ALREADY_FINISHING = "already_finishing"
    NOT_RUNNING = "not_running"
    UNKNOWN_RUN = "unknown_run"
    FORBIDDEN = "forbidden"


class CancelResult(BaseModel):
    """Stop-request disposition, not a claim that execution has stopped.

    run_id is the resolved target (or the explicit unknown target), and may be
    None when there is no run identity. stop_reason is the winning requested
    reason, or the retained terminal reason; refusals need not have a reason.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: CancelOutcome
    run_id: str | None = None
    stop_reason: RunStopReason | None = None


class RunStatus(enum.StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentAvailability(enum.StrEnum):
    RUNNING = "running"
    FINALIZING = "finalizing"
    IDLE = "idle"
    EVICTED = "evicted"


class ReleaseAgentOutcome(enum.StrEnum):
    RELEASED = "released"
    EVICTED = "evicted"


@dataclass(frozen=True, slots=True)
class AgentEviction:
    agent_id: str
    run_id: str
    reason: Literal["ttl", "idle_cap"]
    idle_duration_seconds: float
    root_generation: int


@dataclass(slots=True)
class AgentSummary:
    agent_id: str
    profile: str
    availability: AgentAvailability
    current_run_id: str | None
    current_run_status: RunStatus | None
    turns_used: int | None = None
    initial_task_summary: str | None = None
    current_task_summary: str | None = None
    idle_seconds: float | None = None
    run_elapsed_seconds: float | None = None
    latest_run_id: str | None = None
    ttl_remaining_seconds: float | None = None
    effective_model: str | None = None
    """Configured model alias committed for this retained agent."""
    base_model: str | None = None
    """Canonical committed base model for this retained agent."""
    active_provider: str | None = None
    """Provider serving the retained agent's currently committed deployment."""
    effective_thinking: str | None = None
    """Semantic effective thinking level, not provider wire-level encoding."""
    result_expired: bool = False
    """Whether the retained current-run result has expired."""
    last_run_status: RunStatus | None = None
    """Terminal outcome of the most recently finalized run, if any."""
    context_tokens: int | None = None
    context_window: int | None = None
    compacting: bool = False
    stop_reason: RunStopReason | None = None
    """Current/latest run's structured stop reason, retained after finalization."""


class EmptySubagentResponseError(ValueError):
    """A completed subagent run produced no substantive response."""


class SubagentManagementError(ValueError):
    """Base error for a retained subagent lifecycle operation."""


class AgentEvictedError(SubagentManagementError):
    """The agent was evicted and cannot be reused."""


class AgentBusyError(SubagentManagementError):
    """The agent is busy with another run."""


class AgentProfileMismatchError(SubagentManagementError):
    """The requested profile does not match the retained agent."""


class AgentResultExpiredError(SubagentManagementError):
    """The retained result has expired."""


class UnknownAgentError(SubagentManagementError):
    """The agent or run does not exist."""


class LaunchConfigError(SubagentManagementError):
    """Base error for rejected launch configuration with a safe field path."""

    def __init__(
        self, field: str, message: str = "Invalid launch configuration"
    ) -> None:
        self.field = field
        self.field_path = field
        super().__init__(f"{message} at {field}")


class InvalidLaunchConfigError(LaunchConfigError):
    """A supplied launch configuration field is invalid."""


class InvalidLaunchModelError(InvalidLaunchConfigError):
    """The selected model alias is unknown or not permitted."""


class InvalidLaunchThinkingError(InvalidLaunchConfigError):
    """The selected thinking level is not supported by the model."""


class InvalidLaunchPromptError(InvalidLaunchConfigError):
    """The selected system prompt is unavailable."""


class InvalidLaunchToolError(InvalidLaunchConfigError):
    """A launch tool selection or override is invalid."""


class ImmutableLaunchPersonaError(InvalidLaunchConfigError):
    """A retained agent's frozen persona was changed."""


class MissingAgentProfileError(LaunchConfigError):
    """The selected launch profile is unavailable."""


class UnsupportedChildForkError(LaunchConfigError):
    """Forking a child launch context is intentionally unsupported."""


class SubagentManagementPort(Protocol):
    """Port for managing background subagent lifecycle."""

    async def check_agents(self) -> list[AgentSummary]:
        """List all retained agents and their current run status. Nonblocking."""
        ...

    async def get_agent_result(
        self, agent_id: str, run_id: str | None = None
    ) -> TaskResult | None:
        """Nonblocking retrieval of a completed run's result.

        If run_id is None, returns the latest run's result.
        Returns None while the run is still in progress. Raises UnknownAgentError
        if the agent or run is unknown and AgentResultExpiredError if its result expired.
        """
        ...

    async def wait_for_agent(
        self, agent_id: str, run_id: str | None = None, *, timeout: float | None = None
    ) -> TaskResult:
        """Block until the specified run completes, with optional timeout.

        Timeout must NOT cancel the background run — it only unblocks the waiter.
        Raises TimeoutError if the timeout expires before completion.
        Raises ValueError for invalid agent_id, run_id mismatch, or non-positive timeout.
        """
        ...

    async def cancel_run(
        self,
        agent_id: str,
        run_id: str | None = None,
        *,
        reason: RunStopReason,
        requester_session_id: str,
    ) -> CancelResult:
        """Request a run-pinned stop without releasing identity or wait leases."""
        ...

    async def release_agent(self, agent_id: str) -> ReleaseAgentOutcome:
        """Close and remove a retained agent or evicted tombstone.

        Returns RELEASED for a live agent and EVICTED when removing its tombstone.
        Raises ValueError for unknown agent_id.
        """
        ...


class SubagentRunnerPort(Protocol):
    def run(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]: ...


@dataclass(slots=True)
class SubagentRunAccumulator:
    _response: list[str] = field(default_factory=list)
    _completed: bool = True

    def observe(self, event: BaseEvent, *, tool_call_id: str) -> ToolStreamEvent | None:
        if isinstance(event, AssistantEvent):
            if event.content:
                self._response.append(event.content)
            if event.stopped_by_middleware:
                self._completed = False
            return None
        if not isinstance(event, ToolResultEvent):
            return None
        if event.skipped:
            self._completed = False
            return None
        if event.result is None or event.tool_class is None:
            return None
        if event.presentation is not None:
            display = event.presentation.display
        else:
            display = ToolUIDataAdapter(event.tool_class).get_result_display(event)
        return ToolStreamEvent(
            tool_name="task",
            message=f"{event.tool_name}: {display.text}",
            tool_call_id=tool_call_id,
        )

    def record_error(self, message: str) -> None:
        self._completed = False
        self._response.append(f"\n[Subagent error: {message}]")

    def build_result(self, *, turns_used: int, completed: bool = True) -> TaskResult:
        completed = self._completed and completed
        response = "".join(self._response)
        if completed and not response.strip():
            raise EmptySubagentResponseError(
                "Completed subagent produced an empty response"
            )
        return TaskResult(response=response, turns_used=turns_used, completed=completed)


def prepare_subagent_prompt(task: str, ctx: InvokeContext) -> str:
    if ctx.scratchpad_dir is None:
        return task
    return (
        f"Scratchpad directory: {ctx.scratchpad_dir}\n"
        "You can read and write files here without permission prompts.\n\n"
        f"{task}"
    )
