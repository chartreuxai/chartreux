from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

from pydantic import BaseModel

from chartreux.core.agent_loop._execution_admission import ExecutionAdmission
from chartreux.core.config._restrictions import SourceRestrictions
from chartreux.core.hooks.models import HookConfigResult
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.manager import ToolManager
from chartreux.core.usage import AsyncUsageWriter
from chartreux.core.utils import RetryObserver
from chartreux.core.workspace import Workspace
from chartreux.utils.cache_store import CacheStore


class ToolExecutionResponse(StrEnum):
    SKIP = auto()
    EXECUTE = auto()


class ToolDecision(BaseModel):
    verdict: ToolExecutionResponse
    approval_type: ToolPermission
    feedback: str | None = None
    result_note: str | None = None
    # Invocation-bound authority the execution-admission gate revalidates at
    # the handoff (after the snapshot await, immediately before invoke). Only
    # an "Allow once" approval carries one: its permit is consumed before the
    # call resumes, so the gate re-arms it for exactly this invocation.
    admission: ExecutionAdmission | None = None


@dataclass(frozen=True, slots=True)
class RootAccountingOwner:
    """Immutable root identity and shared writer, independent of transcript logging.

    Session bindings capture attribution per invocation. Writer retirement belongs
    to the runtime, after all producers sharing this owner have settled.
    """

    root_session_id: str
    project_key: str
    writer: AsyncUsageWriter
    # Runtime-owned allocation seam for clear/rewind, which mint root identities.
    owner_factory: Callable[[str, str], RootAccountingOwner] | None = None


@dataclass(frozen=True, slots=True)
class AgentRuntimePolicy:
    max_turns: int | None
    max_price: float | None
    max_tokens: int | None
    max_session_tokens: int | None
    enable_streaming: bool
    launch_context: dict[str, object] | None
    headless: bool
    hook_config_result: HookConfigResult | None
    cache_store: CacheStore
    user_input_capability: bool = False
    # Whether this surface wants background LLM session titles. Core owns the
    # capability; the delivery layer (app server) owns this policy decision.
    auto_title_enabled: bool = False
    inherited_workspace: Workspace | None = None
    inherited_restrictions: tuple[SourceRestrictions, ...] = ()
    inherited_mode_restrictions: tuple[SourceRestrictions, ...] = ()
    inherited_plan_write_scopes: tuple[tuple[Path, Path | None], ...] = ()
    # Bare canonical roots granted to the launching session at dispatch time.
    # A dedicated slot, never a synthesized config-source authority, so policy
    # replacement and partitioning never treat session grants as policy.
    inherited_root_grants: tuple[Path, ...] = ()
    # Per-run budget maxes (anchor-relative to the run-admission baseline).
    # None means unconfigured: the middleware is omitted and replay admission
    # ignores the run budget.
    run_max_price: float | None = None
    run_max_session_tokens: int | None = None
    parent_authority_getter: Callable[[], ToolManager] | None = None
    parent_authority_revision_getter: Callable[[], int] | None = None


@dataclass(frozen=True, slots=True)
class AgentTurnOptions:
    turn_id: str | None = None
    retry_sink: RetryObserver | None = None
    injected: bool = False
    user_initiated_retry: bool = False
