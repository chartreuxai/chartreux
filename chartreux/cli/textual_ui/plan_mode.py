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

from collections.abc import Awaitable, Callable, Mapping, Sequence
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
from chartreux.utils.io import decode_safe

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
    """The allocated plan workspace: its task directory and its document.

    ``episode_id`` and ``sidecar_path`` are the allocation's exclusive episode
    claim (``core.plan_mode.allocate_plan_workspace``). ``None`` means the
    workspace was not claimed as an episode — the legacy shape, which accept
    never arms on.
    """

    directory: Path
    document_path: Path
    episode_id: str | None = None
    sidecar_path: Path | None = None


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
    """The persisted record for an entered mode (design §7).

    An episode-bound workspace carries its identity across so a resume can
    consult the sidecar first; the entry record is a draft (the scan domain
    mirror belongs to the P1 claim, which only the accept transaction makes).
    """
    episode_id = state.workspace.episode_id
    return SessionPlanRecord(
        plan_path=state.plan_path,
        theme=state.theme,
        planner_model=state.planner.model,
        planner_thinking=state.planner.thinking,
        pre_entry_active_model=state.pre_entry_active_model,
        pre_entry_planner_thinking=state.pre_entry_planner_thinking,
        episode_id=episode_id,
        sidecar_path=(
            None
            if state.workspace.sidecar_path is None
            else str(state.workspace.sidecar_path)
        ),
        draft_path=state.plan_path if episode_id is not None else None,
        state=EPISODE_STATE_DRAFT if episode_id is not None else None,
    )


def state_from_record(record: SessionPlanRecord) -> PlanModeState:
    """Restore the in-memory mode from a persisted record.

    The bound draft is authoritative for the episode's DOCUMENT; a legacy
    record only ever names ``plan_path``. The episode's workspace directory is
    the SIDECAR's directory, never the document's parent: allocation wrote the
    sidecar, the snapshot, and the fence into the allocated episode directory,
    and a re-bind moves only the document. A record with no sidecar (legacy)
    falls back to the document's parent, where allocation put the document.
    """
    document_path = Path(record.draft_path or record.plan_path)
    sidecar = None if record.sidecar_path is None else Path(record.sidecar_path)
    return PlanModeState(
        workspace=PlanWorkspacePaths(
            directory=document_path.parent if sidecar is None else sidecar.parent,
            document_path=document_path,
            episode_id=record.episode_id,
            sidecar_path=(
                None if record.sidecar_path is None else Path(record.sidecar_path)
            ),
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
            directory=workspace.directory,
            document_path=workspace.document_path,
            episode_id=workspace.episode_id,
            sidecar_path=workspace.sidecar_path,
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


# --- Episode surfaces (design §1, §4.2, §4.4, §4.5) -------------------------
#
# The sidecar is the single arbiter; these names mirror core.plan_episode's
# recovery rows. The TUI builders stay free of a static core import, and
# test_plan_episode_flow pins the mirrors against the core constants.

EPISODE_ROW_DRAFT = "draft"
EPISODE_ROW_ACCEPTING_NO_SNAPSHOT = "accepting-no-snapshot"
EPISODE_ROW_ACCEPTING_SNAPSHOT = "accepting-snapshot"
EPISODE_ROW_ACCEPTED = "accepted"
EPISODE_ROW_CANCELLED = "cancelled"
EPISODE_ROW_EXECUTED = "executed"
EPISODE_ROW_POSSIBLY_EXECUTED = "possibly-executed"
EPISODE_ROW_UNRESOLVABLE = "unresolvable"
EPISODE_ROW_LEGACY = "legacy-unverified"

EPISODE_STATE_DRAFT = "draft"
EPISODE_STATE_ACCEPTED = "accepted"
EPISODE_STATE_CANCELLED = "cancelled"

TERMINAL_EPISODE_ROWS = frozenset({EPISODE_ROW_ACCEPTED, EPISODE_ROW_CANCELLED})
RECOVERABLE_EPISODE_ROWS = frozenset({
    EPISODE_ROW_ACCEPTING_NO_SNAPSHOT,
    EPISODE_ROW_ACCEPTING_SNAPSHOT,
})
POSSIBLY_EXECUTED_EPISODE_ROWS = frozenset({
    EPISODE_ROW_POSSIBLY_EXECUTED,
    EPISODE_ROW_EXECUTED,
})

_START_NEW_ACTION = "/plan to start a new plan"
_CANCEL_ACTION = "/plan cancel to cancel"
_PIN_RESTORE_ACTION = "/plan cancel to cancel with the pre-entry selections restored"
_REBIND_ACTION = "/plan rebind <path> to bind a fresh episode"
_LEGACY_REBIND_ACTION = (
    "/plan rebind <path> to bind a fresh episode to an authorized document"
)
_ABORT_ACTION = "/plan cancel aborts the claim and returns the episode to draft"
_MARK_ACCEPTED_ACTION = (
    "/plan mark-accepted marks the episode accepted; its handoff is foreclosed, "
    "re-execution of the same episode is refused, and the remedy is a fresh "
    "episode re-binding the surviving proof path"
)


@dataclass(frozen=True, slots=True)
class LoggingDomainView:
    """The session's ``{save_dir, session_prefix}`` logging domain."""

    save_dir: str
    session_prefix: str


@dataclass(frozen=True, slots=True)
class EpisodeView:
    """One sidecar-first observation, in TUI-owned terms."""

    row: str
    state: str | None = None
    episode_id: str | None = None
    owner: bool = False
    executor_session_id: str | None = None
    snapshot_name: str | None = None
    record_mismatch: bool = False
    rename_outcome: str | None = None

    @property
    def draft(self) -> bool:
        """A draft episode is the one resume state that continues as active."""
        return self.row == EPISODE_ROW_DRAFT

    @property
    def terminal(self) -> bool:
        """An accepted or cancelled episode is history; only a new one continues."""
        return self.row in TERMINAL_EPISODE_ROWS

    @property
    def recoverable(self) -> bool:
        """An accepting episode can be re-offered or aborted by its owner."""
        return self.row in RECOVERABLE_EPISODE_ROWS

    @property
    def possibly_executed(self) -> bool:
        """An executor records the episode; it is never auto-retried."""
        return self.row in POSSIBLY_EXECUTED_EPISODE_ROWS

    @property
    def legacy(self) -> bool:
        return self.row == EPISODE_ROW_LEGACY

    @property
    def unresolvable(self) -> bool:
        return self.row == EPISODE_ROW_UNRESOLVABLE


@dataclass(frozen=True, slots=True)
class ConfirmedPlan:
    """The bytes the consent modal last showed and the user confirmed."""

    content: bytes
    sha256: str
    summary: str


def session_logging_section(
    fields: Sequence[ConfigFieldWire],
) -> Mapping[str, object] | None:
    """The effective ``[session_logging]`` section, or None when unreadable."""
    value = _overrides_agnostic_value(fields, "session_logging")
    return value if isinstance(value, Mapping) else None


def session_logging_enabled(fields: Sequence[ConfigFieldWire]) -> bool | None:
    """The section's enabled flag; None when the projection does not carry it."""
    section = session_logging_section(fields)
    if section is None:
        return None
    enabled = section.get("enabled")
    return enabled if isinstance(enabled, bool) else None


def logging_entry_guard(enabled: bool) -> str | None:
    """The ``/plan`` entry and resume guard; the refusal detail or None."""
    plan_episode = _core_plan_episode()
    try:
        plan_episode.require_session_logging(enabled)
    except plan_episode.PlanLoggingDisabledError as error:
        return str(error)
    return None


def logging_domain(fields: Sequence[ConfigFieldWire]) -> LoggingDomainView | None:
    """The session's own logging domain, or None when it cannot be resolved."""
    section = session_logging_section(fields)
    if section is None or section.get("enabled") is False:
        return None
    save_dir = section.get("save_dir")
    session_prefix = section.get("session_prefix")
    if not isinstance(save_dir, str) or not save_dir:
        return None
    if not isinstance(session_prefix, str) or not session_prefix:
        return None
    return LoggingDomainView(save_dir=save_dir, session_prefix=session_prefix)


def default_workspaces_root() -> Path:
    """The chartreux-owned plan workspaces root (a rebind authority)."""
    return _core_plan_mode().default_workspaces_root()


def episode_directory(record: SessionPlanRecord) -> Path | None:
    """The episode directory a record points at; None for a legacy record."""
    if record.sidecar_path:
        return Path(record.sidecar_path).parent
    if record.draft_path:
        return Path(record.draft_path).parent
    if record.episode_id:
        return Path(record.plan_path).parent
    return None


def episode_view_from_state(state: PlanModeState) -> EpisodeView:
    """The benign draft view of an armed mode (no claim is held)."""
    return EpisodeView(
        row=EPISODE_ROW_DRAFT,
        state=EPISODE_STATE_DRAFT,
        episode_id=state.workspace.episode_id,
    )


def observe_episode_view(
    record: SessionPlanRecord, *, session_id: str, domain: LoggingDomainView | None
) -> tuple[EpisodeView | None, str | None]:
    """Observe the record's episode sidecar-first; the view or the failure detail.

    A record with no episode directory has nothing to observe (legacy), and an
    untrusted scan raises out of the core arbiter: both are reported, never
    read as a clean miss.
    """
    directory = episode_directory(record)
    if directory is None:
        return (
            EpisodeView(
                row=EPISODE_ROW_LEGACY,
                state=record.display_state,
                episode_id=record.episode_id,
            ),
            None,
        )
    plan_episode = _core_plan_episode()
    try:
        observation = plan_episode.observe_episode(
            directory,
            record=_core_record(record),
            session_id=session_id,
            owner_domain=_core_domain(domain),
        )
    except plan_episode.PlanEpisodeError as error:
        return None, str(error)
    return (
        EpisodeView(
            row=observation.row,
            state=observation.state,
            episode_id=observation.episode_id,
            owner=observation.owner,
            executor_session_id=observation.executor_session_id,
            snapshot_name=observation.snapshot_name,
            record_mismatch=observation.record_mismatch,
            rename_outcome=observation.rename_outcome,
        ),
        None,
    )


def episode_status_lines(
    view: EpisodeView, *, plan_path: str | None = None, remedy_path: str | None = None
) -> str:
    """The episode's status line and its design-named actions.

    One builder serves the resume banner and ``/plan status`` so the two can
    never drift.
    """
    lines = [
        f"Plan episode {view.episode_id or '(unidentified)'}: {_state_label(view)}."
    ]
    if plan_path:
        lines.append(f"Plan document: {plan_path}.")
    lines.append(f"Actions: {'; '.join(episode_actions(view))}.")
    if view.record_mismatch:
        lines.append(
            "The record's episode does not match the sidecar's; the sidecar is "
            "authoritative."
        )
    if remedy_path is not None:
        lines.append(
            f"The surviving proof path is `{remedy_path}`; re-bind it into a "
            f"fresh episode with `/plan rebind {remedy_path}`."
        )
    return "\n".join(lines)


def episode_actions(view: EpisodeView) -> tuple[str, ...]:  # noqa: PLR0911
    """The design-named actions available in one observed state."""
    row = view.row
    if row == EPISODE_ROW_DRAFT:
        return ("continue planning", "/plan accept to execute", _CANCEL_ACTION)
    if row in RECOVERABLE_EPISODE_ROWS:
        if view.owner:
            return (
                "/plan accept re-offers the accept transaction (owner session only)",
                _ABORT_ACTION,
            )
        holder = "another session"
        return (_START_NEW_ACTION, f"the claim is held by {holder}; inspect it")
    if row in TERMINAL_EPISODE_ROWS:
        return (_START_NEW_ACTION, _PIN_RESTORE_ACTION)
    if row == EPISODE_ROW_LEGACY:
        return (_LEGACY_REBIND_ACTION, _PIN_RESTORE_ACTION)
    if row == EPISODE_ROW_UNRESOLVABLE:
        return (_REBIND_ACTION, _START_NEW_ACTION)
    if row == EPISODE_ROW_EXECUTED:
        return (_START_NEW_ACTION, _PIN_RESTORE_ACTION)
    if row in POSSIBLY_EXECUTED_EPISODE_ROWS:
        return (
            "inspect-executor: open the recorded executor session before deciding",
            _MARK_ACCEPTED_ACTION,
        )
    return ("inspect the episode status",)


def _state_label(view: EpisodeView) -> str:  # noqa: PLR0911
    row = view.row
    if row == EPISODE_ROW_DRAFT:
        return "draft — planning continues"
    if row in RECOVERABLE_EPISODE_ROWS:
        holder = "the owning session" if view.owner else "another session"
        snapshot = (
            "its consent snapshot survives"
            if row == EPISODE_ROW_ACCEPTING_SNAPSHOT
            else "no consent snapshot survives"
        )
        return (
            f"accepting — an accept transaction did not complete ({holder}; {snapshot})"
        )
    if row == EPISODE_ROW_ACCEPTED:
        return "accepted (terminal) — history; it can never be re-accepted or resumed"
    if row == EPISODE_ROW_CANCELLED:
        return "cancelled (terminal) — history; it can never be resumed"
    if row == EPISODE_ROW_POSSIBLY_EXECUTED:
        executor = view.executor_session_id or "(unrecorded)"
        return f"possibly executed — executor session {executor}"
    if row == EPISODE_ROW_EXECUTED:
        executor = view.executor_session_id or "(unrecorded)"
        return f"executed — recorded executor session {executor}"
    if row == EPISODE_ROW_UNRESOLVABLE:
        return (
            "unresolvable — its episode sidecar is missing and no executor records it"
        )
    return "legacy-unverified — the record predates episode identity, so accept is disabled"


def read_document_bytes(path: Path) -> tuple[bytes | None, str | None]:
    """Read the bound draft's RAW bytes (the consent authority)."""
    plan_episode = _core_plan_episode()
    try:
        return plan_episode.read_draft_bytes(path), None
    except plan_episode.PlanDraftReadError as error:
        return None, str(error)


def digest_bytes(content: bytes) -> str:
    """The consent digest over raw bytes, never a decoded text."""
    return _core_plan_episode().sha256_bytes(content)


def brief_from_bytes(path: Path, content: bytes) -> PlanDocumentBrief:
    """The display brief for the modal: the decode, not the consent bytes."""
    plan_mode = _core_plan_mode()
    text = decode_safe(content).text
    return PlanDocumentBrief(
        path=path, content=text, summary=plan_mode.extract_plan_summary(text) or ""
    )


def episode_snapshot_path(directory: Path, episode_id: str) -> Path:
    """The episode's live consent-snapshot name (the handoff's locator)."""
    return _core_plan_episode().accepted_snapshot_path(directory, episode_id)


def abort_accept_claim(
    directory: Path, *, session_id: str, domain: LoggingDomainView
) -> str | None:
    """Abort a held accept claim (cancel during accepting); the failure or None."""
    plan_episode = _core_plan_episode()
    try:
        plan_episode.abort_accept(
            directory, session_id=session_id, owner_domain=_core_domain_required(domain)
        )
    except plan_episode.PlanEpisodeError as error:
        return str(error)
    return None


def cancel_episode(
    directory: Path, *, session_id: str, domain: LoggingDomainView
) -> str | None:
    """The terminal cancel of a draft episode; the failure detail or None."""
    plan_episode = _core_plan_episode()
    try:
        plan_episode.cancel_episode(
            directory, session_id=session_id, owner_domain=_core_domain_required(domain)
        )
    except plan_episode.PlanEpisodeError as error:
        return str(error)
    return None


def release_crashed_claim(
    directory: Path, *, episode_id: str, session_id: str, domain: LoggingDomainView
) -> str | None:
    """Row 2: release a crashed attempt that left no snapshot; the failure or None."""
    plan_episode = _core_plan_episode()
    try:
        plan_episode.recover_release_claim(
            directory,
            episode_id=episode_id,
            session_id=session_id,
            owner_domain=_core_domain_required(domain),
        )
    except plan_episode.PlanEpisodeError as error:
        return str(error)
    return None


async def reopen_accepting_episode(
    directory: Path,
    *,
    record: SessionPlanRecord,
    session_id: str,
    domain: LoggingDomainView,
    write_record: Callable[[SessionPlanRecord], Awaitable[None]],
) -> tuple[RecoveryOfferView | None, str | None]:
    """Row 3: verify the claim and the surviving bytes, then restore the record."""
    plan_episode = _core_plan_episode()

    async def _write(restored: object) -> None:
        await write_record(_wire_record(restored))

    try:
        offer = await plan_episode.reopen_accepting_episode(
            directory,
            episode_id=record.episode_id or "",
            session_id=session_id,
            owner_domain=_core_domain_required(domain),
            record=_core_record(record),
            write_record=_write,
        )
    except plan_episode.PlanEpisodeError as error:
        return None, str(error)
    return (
        RecoveryOfferView(
            episode_id=offer.sidecar.episode_id,
            snapshot_path=offer.snapshot_path,
            content=offer.content,
            consent_sha256=offer.sidecar.consent_sha256 or "",
        ),
        None,
    )


async def cancel_accepting_episode(
    directory: Path,
    *,
    record: SessionPlanRecord,
    session_id: str,
    domain: LoggingDomainView,
    write_record: Callable[[SessionPlanRecord], Awaitable[None]],
) -> str | None:
    """Row 3's cancel leg (re-scan first); the failure detail or None."""
    plan_episode = _core_plan_episode()

    async def _write(restored: object) -> None:
        await write_record(_wire_record(restored))

    try:
        await plan_episode.recover_cancel_accepting(
            directory,
            episode_id=record.episode_id or "",
            session_id=session_id,
            owner_domain=_core_domain_required(domain),
            record=_core_record(record),
            write_record=_write,
        )
    except plan_episode.PlanEpisodePossiblyExecutedError as error:
        return f"{error} (executor session {error.executor_session_id})"
    except plan_episode.PlanEpisodeError as error:
        return str(error)
    return None


def adjudicate_possibly_executed(
    directory: Path, *, episode_id: str, session_id: str, domain: LoggingDomainView
) -> tuple[str | None, str | None]:
    """Mark-accepted: the owner flip to accepted; (rename outcome, failure)."""
    plan_episode = _core_plan_episode()
    try:
        committed = plan_episode.adjudicate_possibly_executed(
            directory,
            episode_id=episode_id,
            session_id=session_id,
            owner_domain=_core_domain_required(domain),
        )
    except plan_episode.PlanEpisodeError as error:
        return None, str(error)
    return committed.rename_outcome, None


def surviving_snapshot_path(record: SessionPlanRecord) -> Path | None:
    """The episode's surviving snapshot by EITHER name (the remedy's path)."""
    directory = episode_directory(record)
    if directory is None or record.episode_id is None:
        return None
    plan_episode = _core_plan_episode()
    sidecar = plan_episode.read_sidecar(directory)
    if sidecar is None:
        return None
    name = plan_episode.surviving_snapshot_name(directory, sidecar)
    return None if name is None else directory / name


def rebind_current_episode(
    state: PlanModeState,
    draft_path: Path,
    *,
    authorized_roots: Sequence[Path],
    session_id: str,
    domain: LoggingDomainView,
) -> tuple[PlanWorkspacePaths | None, str | None]:
    """Re-bind the armed episode's draft path (canonicalized, authorized)."""
    plan_episode = _core_plan_episode()
    directory = state.workspace.sidecar_path
    directory = state.workspace.directory if directory is None else directory.parent
    try:
        canonical = plan_episode.canonicalize_authorized_path(
            draft_path, authorized_roots
        )
        plan_episode.rebind_draft(
            directory,
            draft_path=canonical,
            authorized_roots=authorized_roots,
            session_id=session_id,
            owner_domain=_core_domain_required(domain),
        )
    except plan_episode.PlanEpisodeError as error:
        return None, str(error)
    return (
        PlanWorkspacePaths(
            directory=directory,
            document_path=canonical,
            episode_id=state.workspace.episode_id,
            sidecar_path=state.workspace.sidecar_path,
        ),
        None,
    )


def rebind_fresh_episode(
    draft_path: Path,
    *,
    authorized_roots: Sequence[Path],
    session_id: str,
    domain: LoggingDomainView,
    project_root: Path,
) -> tuple[PlanWorkspacePaths | None, str | None]:
    """Rebind-to-new-episode: a FRESH allocation binding the chosen draft.

    The fresh directory is allocated first (its own exclusive episode claim),
    then its sidecar's draft path is atomically re-bound to the canonicalized
    target under the transition fence. The mandatory re-preview and re-consent
    gate the bytes afterwards.
    """
    plan_episode = _core_plan_episode()
    plan_mode = _core_plan_mode()
    try:
        canonical = plan_episode.canonicalize_authorized_path(
            draft_path, authorized_roots
        )
        workspace = plan_mode.allocate_plan_workspace(
            session_id, project_root=project_root
        )
        plan_episode.rebind_draft(
            workspace.directory,
            draft_path=canonical,
            authorized_roots=authorized_roots,
            session_id=session_id,
            owner_domain=_core_domain_required(domain),
        )
    except plan_episode.PlanEpisodeError as error:
        return None, str(error)
    return (
        PlanWorkspacePaths(
            directory=workspace.directory,
            document_path=canonical,
            episode_id=workspace.episode_id,
            sidecar_path=workspace.sidecar_path,
        ),
        None,
    )


@dataclass(frozen=True, slots=True)
class RecoveryOfferView:
    """Row 3's verified re-offer: the bytes and the episode's live snapshot."""

    episode_id: str
    snapshot_path: Path
    content: bytes
    consent_sha256: str


def _core_plan_episode() -> ModuleType:
    return import_module("chartreux.core.plan_episode")


def _core_record(record: SessionPlanRecord) -> object:
    session_types = import_module("chartreux.core.session_types")
    return session_types.PlanModeRecord.model_validate(
        record.model_dump(mode="json", by_alias=False)
    )


def _wire_record(record: object) -> SessionPlanRecord:
    return SessionPlanRecord.model_validate(
        record.model_dump(mode="json")  # pyright: ignore[reportAttributeAccessIssue]
    )


def _core_domain(domain: LoggingDomainView | None) -> object | None:
    if domain is None:
        return None
    return _core_plan_episode().LoggingDomain(
        save_dir=domain.save_dir, session_prefix=domain.session_prefix
    )


def _core_domain_required(domain: LoggingDomainView) -> object:
    return _core_plan_episode().LoggingDomain(
        save_dir=domain.save_dir, session_prefix=domain.session_prefix
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
