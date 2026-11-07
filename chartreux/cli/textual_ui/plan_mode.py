"""The TUI half of ``/plan`` mode: mode state, config reads, and pin writes.

``chartreux.core.plan_mode`` owns the plan document, the workspace allocation,
and the seed builders. The textual-UI boundary forbids a static
``chartreux.core`` import (``tests/cli/textual_ui/test_app_server_boundary.py``),
so this module imports core lazily inside the functions that need it and hands
the TUI plain local values: the selection snapshot, the workspace paths, and the
accepted document's summary.

This module also owns the TUI-side facts the commands and the consent modal
need: the effective ``[plan]`` section read from the projected config fields,
the pre-entry session-override snapshot that cancellation restores, and the
per-key override writes that entry, cancel, and ``/clear`` issue.

Every write here is per key on the ``overrides`` layer. A whole-map
``/thinking_overrides`` replace would clobber unrelated aliases' session pins.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType

from chartreux.app_server.config import ModelConfigView
from chartreux.app_server.protocol import (
    ConfigFieldWire,
    ConfigWriteOpWire,
    ProtocolError,
    ProtocolErrorCode,
    SessionPlanRecord,
)

OVERRIDES_LAYER = "overrides"
ACTIVE_MODEL_PATH = "/active_model"
THINKING_OVERRIDES_PATH = "/thinking_overrides"

# The standing indicator: this class tints the prompt area (the amber borders
# and label color live in app.tcss), and the label rides the prompt's border
# title. Entry, resume restore, and the exit paths only apply or drop them.
PLAN_MODE_TINT_CLASS = "plan-mode"
PLAN_MODE_LABEL = "plan"


def _core_plan_mode() -> ModuleType:
    return import_module("chartreux.core.plan_mode")


def _core_config_models() -> ModuleType:
    return import_module("chartreux.core.config.models")


def _core_config_patch() -> ModuleType:
    return import_module("chartreux.core.config.patch")


@dataclass(frozen=True, slots=True)
class PlannerSelection:
    """The ``[plan.planner]`` selection, as ``/plan`` pins it."""

    model: str
    thinking: str


@dataclass(frozen=True, slots=True)
class ExecutorSelection:
    """The ``[plan.executor]`` selection shown at consent and applied at handoff."""

    model: str
    thinking: str
    identity: str


@dataclass(frozen=True, slots=True)
class PlanSelection:
    """Both ``[plan]`` selections in one immutable snapshot."""

    planner: PlannerSelection
    executor: ExecutorSelection


@dataclass(frozen=True, slots=True)
class PlanWorkspacePaths:
    """The allocated plan workspace: its task directory and its document."""

    directory: Path
    document_path: Path


@dataclass(frozen=True, slots=True)
class PlanDocumentBrief:
    """An accepted plan document: where it is and what it says."""

    path: Path
    content: str
    summary: str


@dataclass(frozen=True, slots=True)
class PlanModeState:
    """The in-memory plan mode: the workspace and the restore snapshot.

    ``pre_entry_*`` hold the overrides-layer values as they stood before
    ``/plan`` wrote its pins. ``None`` means the key was absent, which cancel
    expresses as a per-key removal rather than a restore.
    """

    workspace: PlanWorkspacePaths
    planner: PlannerSelection
    pre_entry_active_model: str | None
    pre_entry_planner_thinking: str | None
    theme: str | None = None

    @property
    def plan_path(self) -> str:
        return str(self.workspace.document_path)


def plan_section(fields: Sequence[ConfigFieldWire]) -> PlanSelection:
    """The effective ``[plan]`` selections carried by the config fields.

    Raises ``ValueError`` when the projected value is not a readable section:
    the caller reports it instead of displaying selections the server would
    refuse at handoff.
    """
    models = _core_config_models()
    plan_mode = _core_plan_mode()
    value = _overrides_agnostic_value(fields, "plan")
    section = (
        models.PlanSectionConfig()
        if not isinstance(value, Mapping)
        else models.PlanSectionConfig.model_validate(dict(value))
    )
    planner = plan_mode.planner_selection(section)
    executor = plan_mode.executor_selection(section)
    return PlanSelection(
        planner=PlannerSelection(model=planner.model, thinking=planner.thinking),
        executor=ExecutorSelection(
            model=executor.model, thinking=executor.thinking, identity=executor.identity
        ),
    )


def pre_entry_snapshot(
    fields: Sequence[ConfigFieldWire],
    planner_model: str,
    models: Sequence[ModelConfigView],
) -> tuple[str | None, str | None]:
    """The overrides layer's ``active_model`` and planner-alias thinking values."""
    active_model = _overrides_layer_value(fields, "active_model")
    return (
        active_model if isinstance(active_model, str) and active_model else None,
        _planner_thinking_override(fields, planner_model, models),
    )


def entry_ops(planner: PlannerSelection) -> list[ConfigWriteOpWire]:
    """The planner model pin and the planner-alias thinking override."""
    return [
        _set(ACTIVE_MODEL_PATH, planner.model),
        _set(thinking_path(planner.model), planner.thinking),
    ]


def restore_ops(state: PlanModeState) -> list[ConfigWriteOpWire]:
    """Restore the pre-entry snapshot per key.

    Shared by ``/plan cancel`` and the best-effort revert of a failed entry:
    a key that existed pre-entry is set back to its value, a key that did not
    exist is removed.
    """
    active_model = (
        _set(ACTIVE_MODEL_PATH, state.pre_entry_active_model)
        if state.pre_entry_active_model is not None
        else _remove(ACTIVE_MODEL_PATH)
    )
    planner_thinking = (
        _set(thinking_path(state.planner.model), state.pre_entry_planner_thinking)
        if state.pre_entry_planner_thinking is not None
        else _remove(thinking_path(state.planner.model))
    )
    return [active_model, planner_thinking]


def clear_ops(state: PlanModeState) -> list[ConfigWriteOpWire]:
    """Remove every pin ``/plan`` wrote; used by ``/clear`` while planning.

    ``/clear`` voids the saved restore snapshot, so it removes the keys
    unconditionally instead of restoring pre-entry values.
    """
    return [_remove(ACTIVE_MODEL_PATH), _remove(thinking_path(state.planner.model))]


def record_from_state(state: PlanModeState) -> SessionPlanRecord:
    """The persisted record for an entered mode (design §7)."""
    return SessionPlanRecord(
        plan_path=state.plan_path,
        theme=state.theme,
        planner_model=state.planner.model,
        planner_thinking=state.planner.thinking,
        pre_entry_active_model=state.pre_entry_active_model,
        pre_entry_planner_thinking=state.pre_entry_planner_thinking,
    )


def state_from_record(record: SessionPlanRecord) -> PlanModeState:
    """Restore the in-memory mode from a persisted record.

    The workspace directory is derived from the document path: allocation put
    the document inside the task directory, and the record only ever names
    the document.
    """
    document_path = Path(record.plan_path)
    return PlanModeState(
        workspace=PlanWorkspacePaths(
            directory=document_path.parent, document_path=document_path
        ),
        planner=PlannerSelection(
            model=record.planner_model, thinking=record.planner_thinking
        ),
        pre_entry_active_model=record.pre_entry_active_model,
        pre_entry_planner_thinking=record.pre_entry_planner_thinking,
        theme=record.theme,
    )


def resume_notice(record: SessionPlanRecord) -> str:
    """The one-line resume notice, with the theme clause only when themed."""
    notice = f"Plan mode resumed. Plan document: {record.plan_path}."
    if record.theme:
        return f"{notice} Theme: {record.theme}."
    return notice


def plan_status(plan_path: str, theme: str | None) -> str:
    """The ``/plan status`` line: mode, theme, and plan document.

    The wording belongs to the core builder the compaction boundary's
    re-injected notice uses, so the status line and what the model is told
    after a compaction cannot drift apart.
    """
    return _core_plan_mode().build_plan_mode_compaction_notice(plan_path, theme)


def thinking_path(alias: str) -> str:
    escape = _core_config_patch().escape_json_pointer_token
    return f"{THINKING_OVERRIDES_PATH}/{escape(alias)}"


def handoff_failed_before_replacement(error: ProtocolError) -> bool:
    """Whether a failed handoff provably left the planner conversation in place.

    The server refuses an active turn or background job with a bare CONFLICT
    before it writes anything. Every other failure is at or after the reset,
    where the fresh executor session may already be bound — including the
    post-bind apply-mismatch CONFLICT, which carries the applied values.
    """
    return error.code is ProtocolErrorCode.CONFLICT and error.data is None


def allocate_workspace(
    session_id: str, *, project_root: Path
) -> tuple[PlanWorkspacePaths | None, str | None]:
    """Allocate the session's plan workspace; the paths or the failure detail."""
    plan_mode = _core_plan_mode()
    try:
        workspace = plan_mode.allocate_plan_workspace(
            session_id, project_root=project_root
        )
    except plan_mode.PlanWorkspaceError as error:
        return None, str(error)
    return (
        PlanWorkspacePaths(
            directory=workspace.directory, document_path=workspace.document_path
        ),
        None,
    )


def release_workspace(workspace: PlanWorkspacePaths) -> bool:
    """Remove the workspace when cancel leaves it empty; True when it went away."""
    plan_mode = _core_plan_mode()
    return plan_mode.remove_empty_plan_workspace(workspace.directory)


def read_document(path: Path) -> tuple[PlanDocumentBrief | None, str | None]:
    """Read the plan document for acceptance; the brief or the failure detail."""
    plan_mode = _core_plan_mode()
    try:
        document = plan_mode.read_plan_document(path)
    except plan_mode.PlanDocumentError as error:
        return None, str(error)
    return (
        PlanDocumentBrief(
            path=document.path, content=document.content, summary=document.summary
        ),
        None,
    )


def planner_seed(workspace: PlanWorkspacePaths, theme: str | None = None) -> str:
    """The visible entry message telling the planner where to write the plan."""
    plan_mode = _core_plan_mode()
    return plan_mode.build_planner_seed(
        plan_mode.PlanWorkspace(
            directory=workspace.directory, document_path=workspace.document_path
        ),
        theme=theme,
    )


def executor_seed(brief: PlanDocumentBrief) -> str:
    """The handoff brief: the displayed summary plus the absolute plan path."""
    plan_mode = _core_plan_mode()
    return plan_mode.build_executor_seed(
        plan_mode.PlanDocument(
            path=brief.path, content=brief.content, summary=brief.summary
        )
    )


def _overrides_agnostic_value(
    fields: Sequence[ConfigFieldWire], name: str
) -> object | None:
    entry = next((field for field in fields if field.name == name), None)
    return entry.value if entry is not None else None


def _overrides_layer_value(
    fields: Sequence[ConfigFieldWire], name: str
) -> object | None:
    """The overrides-layer value of one field, or None when the layer omits it."""
    entry = next((field for field in fields if field.name == name), None)
    if entry is None:
        return None
    return next(
        (layer.value for layer in entry.layer_values if layer.layer == OVERRIDES_LAYER),
        None,
    )


def _planner_thinking_override(
    fields: Sequence[ConfigFieldWire],
    planner_model: str,
    models: Sequence[ModelConfigView],
) -> str | None:
    """The planner alias's pre-entry thinking override, when one was stored.

    Override keys are canonical base models. The configured planner value is
    matched as written, then through the model whose wire name it names, so a
    deployment-name expression still finds its key.
    """
    value = _overrides_layer_value(fields, "thinking_overrides")
    if not isinstance(value, Mapping):
        return None
    overrides = {
        str(key): level for key, level in value.items() if isinstance(level, str)
    }
    candidates = [
        planner_model,
        *(model.alias for model in models if model.name == planner_model),
    ]
    for candidate in candidates:
        level = overrides.get(candidate)
        if level:
            return level
    return None


def _set(path: str, value: str) -> ConfigWriteOpWire:
    return ConfigWriteOpWire(
        op="set", path=path, value=value, target_layer=OVERRIDES_LAYER
    )


def _remove(path: str) -> ConfigWriteOpWire:
    return ConfigWriteOpWire(op="remove", path=path, target_layer=OVERRIDES_LAYER)
