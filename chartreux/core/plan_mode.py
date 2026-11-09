"""Plan-mode document helpers: selections, workspace allocation, reads, seeds.

``/plan`` keeps one authoritative Markdown document per planning session under
the project's workspace ``tasks/`` directory. The planner writes it; the
execution session is seeded with its ``## Summary`` text and its absolute path.
This module owns the path derivation, the non-destructive allocation, and the
document reads shared by the TUI and the app server, plus the immutable
selection snapshots the consent modal displays and the handoff applies.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
import re

from chartreux.config_values import ThinkingLevel
from chartreux.core.config.models import PlanSectionConfig
from chartreux.core.paths import CHARTREUX_HOME
from chartreux.core.plan_episode import (
    RECOVERY_ROW_UNRESOLVABLE,
    EpisodeObservation,
    PlanEpisodeSidecar,
    cleanup_episode_workspace,
    create_sidecar,
    ensure_episode_fence,
    fence_path,
    new_episode_id,
    sidecar_path,
)
from chartreux.core.session.session_id import extract_suffix
from chartreux.utils.io import read_safe
from chartreux.utils.private_paths import create_private_directory

PLAN_DIRECTORY_PREFIX = "plan-"
PLAN_DOCUMENT_FILENAME = "plan.md"
PLAN_SUMMARY_HEADING = "Summary"

# The ownership marker: written only for a directory this module created, so a
# reused convention-matching directory is never claimed. Presence is the
# authority cancel consults — the path convention alone proves nothing.
PLAN_MARKER_FILENAME = ".chartreux-plan-mode"
_PLAN_MARKER_CONTENT = "chartreux plan-mode workspace\n"

# The session-derived name comes first; numbered alternatives follow so an
# existing plan document is never overwritten.
_MAX_NAME_ATTEMPTS = 100

# The summary is exactly the level-2 section, which ends at the next level-1
# or level-2 heading.
_SUMMARY_HEADING_LEVEL = 2
_SECTION_END_LEVEL = 2

_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_ATX_HEADING = re.compile(r" {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*#*[ \t]*")
_FENCE_OPEN = re.compile(r" {0,3}(`{3,}|~{3,})(.*)")
_FENCE_CLOSE = re.compile(r" {0,3}(`{3,}|~{3,})[ \t]*")


class PlanModeError(Exception):
    """Base class for plan-mode workspace and document failures."""


class PlanWorkspaceError(PlanModeError):
    """A plan workspace name could not be derived or allocated."""


class PlanDocumentError(PlanModeError):
    """A plan document cannot be used as the handoff source."""

    def __init__(self, path: Path, detail: str) -> None:
        super().__init__(f"{detail}: {path}")
        self.path = path


class PlanDocumentMissingError(PlanDocumentError):
    def __init__(self, path: Path) -> None:
        super().__init__(path, "plan document does not exist")


class PlanDocumentUnreadableError(PlanDocumentError):
    def __init__(self, path: Path, detail: str = "plan document is not a file") -> None:
        super().__init__(path, detail)


class PlanDocumentEmptyError(PlanDocumentError):
    def __init__(self, path: Path) -> None:
        super().__init__(path, "plan document is empty")


class PlanSummaryMissingError(PlanDocumentError):
    def __init__(self, path: Path) -> None:
        super().__init__(path, "plan document has no '## Summary' section")


class PlanSummaryEmptyError(PlanDocumentError):
    def __init__(self, path: Path) -> None:
        super().__init__(path, "plan document's '## Summary' section is empty")


@dataclass(frozen=True, slots=True)
class PlanWorkspace:
    """The allocated task directory, its document, and the episode it holds."""

    directory: Path
    document_path: Path
    episode_id: str | None = None
    sidecar_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PlanDocument:
    """A plan document that passed the acceptance reads."""

    path: Path
    content: str
    summary: str


@dataclass(frozen=True, slots=True)
class PlanPlannerSelection:
    model: str
    thinking: ThinkingLevel


@dataclass(frozen=True, slots=True)
class PlanExecutorSelection:
    model: str
    thinking: ThinkingLevel
    identity: str


def planner_selection(config: PlanSectionConfig) -> PlanPlannerSelection:
    planner = config.planner
    return PlanPlannerSelection(model=planner.model, thinking=planner.thinking)


def executor_selection(config: PlanSectionConfig) -> PlanExecutorSelection:
    """The executor snapshot the consent modal displays and the handoff applies."""
    executor = config.executor
    return PlanExecutorSelection(
        model=executor.model, thinking=executor.thinking, identity=executor.identity
    )


def default_workspaces_root() -> Path:
    return CHARTREUX_HOME.path / "workspaces"


def derive_project_name(project_root: Path) -> str:
    """Derive the workspace project directory name from the repository root."""
    try:
        name = project_root.resolve().name
    except OSError as error:
        raise PlanWorkspaceError(
            f"cannot resolve project root for the plan workspace: {project_root}"
        ) from error
    if not name or _SAFE_NAME.fullmatch(name) is None:
        raise PlanWorkspaceError(
            f"cannot derive a plan workspace project name from {project_root}"
        )
    return name


def derive_session_suffix(session_id: str) -> str:
    """Derive the stable per-session task directory suffix."""
    suffix = extract_suffix(session_id)
    if _SAFE_NAME.fullmatch(suffix) is None:
        raise PlanWorkspaceError(
            f"cannot derive a plan workspace name from session {session_id!r}"
        )
    return suffix


def allocate_plan_workspace(
    session_id: str, *, project_root: Path, workspaces_root: Path | None = None
) -> PlanWorkspace:
    """Allocate the session's plan workspace and claim it as an episode.

    Uses ``<workspaces_root>/<project>/tasks/plan-<suffix>/plan.md``; when that
    name already holds a plan document OR an episode sidecar, the first free
    numbered alternative (``plan-<suffix>-2``, ...) is allocated instead. The
    sidecar create is EXCLUSIVE and IS the directory claim: a directory still
    holding a draft is never reused, so two sequential entries in one session
    get distinct episodes and directories. The task directory is created
    owner-only; the document itself is left for the planner to write. A
    directory this call created carries :data:`PLAN_MARKER_FILENAME`; a reused
    one keeps whatever it already held, marker or not.
    """
    suffix = derive_session_suffix(session_id)
    project_name = derive_project_name(project_root)
    root = workspaces_root if workspaces_root is not None else default_workspaces_root()
    tasks_root = (root / project_name / "tasks").absolute()
    base = f"{PLAN_DIRECTORY_PREFIX}{suffix}"
    for attempt in range(1, _MAX_NAME_ATTEMPTS + 1):
        directory = tasks_root / (base if attempt == 1 else f"{base}-{attempt}")
        document_path = directory / PLAN_DOCUMENT_FILENAME
        if _episode_claimed(directory, document_path):
            continue
        # Ownership is what the exclusive create reports, never a pre-call
        # exists() test: a concurrent creator can win that interval, and a
        # marker written into its directory would make it look claimable.
        created_here = _create_workspace_directory(directory)
        if not directory.is_dir():
            raise PlanWorkspaceError(
                f"cannot create plan workspace directory: {directory}"
            )
        if _episode_claimed(directory, document_path):
            if created_here:
                _relinquish_marker(directory)
            continue
        sidecar = _claim_episode(directory, document_path, created_here=created_here)
        if sidecar is None:
            continue
        return PlanWorkspace(
            directory=directory,
            document_path=document_path,
            episode_id=sidecar.episode_id,
            sidecar_path=sidecar_path(directory),
        )
    raise PlanWorkspaceError(
        f"no free plan workspace name for session {session_id!r} under {tasks_root}"
    )


def _episode_claimed(directory: Path, document_path: Path) -> bool:
    """The allocation probe: a draft document or an episode sidecar takes the name."""
    return document_path.exists() or sidecar_path(directory).exists()


def _claim_episode(
    directory: Path, document_path: Path, *, created_here: bool
) -> PlanEpisodeSidecar | None:
    """Claim the directory with the exclusive sidecar; None when another won."""
    sidecar = PlanEpisodeSidecar(episode_id=new_episode_id(), draft_path=document_path)
    if created_here:
        # Allocation is the directory plus its marker: a created-but-unmarked
        # directory must never be left behind as foreign content.
        try:
            _write_plan_marker(directory)
        except OSError as error:
            _relinquish_marker(directory)
            raise PlanWorkspaceError(
                f"cannot mark the plan workspace directory: {directory}: {error}"
            ) from error
    try:
        create_sidecar(directory, sidecar)
    except FileExistsError:
        # A concurrent claimer owns this directory now; only our own marker
        # goes, and the foreign sidecar is never touched.
        if created_here:
            _relinquish_marker(directory)
        return None
    except OSError as error:
        _abandon_claim(directory, created_here=created_here)
        raise PlanWorkspaceError(
            f"cannot claim the plan episode directory: {directory}: {error}"
        ) from error
    try:
        ensure_episode_fence(directory)
    except OSError as error:
        _abandon_claim(directory, created_here=created_here)
        raise PlanWorkspaceError(
            f"cannot open the plan episode fence: {directory}: {error}"
        ) from error
    return sidecar


def _abandon_claim(directory: Path, *, created_here: bool) -> None:
    """Undo this call's claim after a later allocation step failed."""
    if created_here:
        _discard_created_directory(directory)
        return
    with suppress(OSError):
        sidecar_path(directory).unlink()


def _create_workspace_directory(directory: Path) -> bool:
    """Create the leaf exclusively; report whether this call created it."""
    try:
        return create_private_directory(directory)
    except OSError as error:
        raise PlanWorkspaceError(
            f"cannot create plan workspace directory: {directory}: {error}"
        ) from error


def remove_empty_plan_workspace(directory: Path) -> bool:
    """Release an allocated plan workspace, removing only plan-owned entries.

    Ownership is the marker file, not the path convention: a reused,
    convention-matching directory that carries no marker is foreign and is never
    touched. The removal whitelist (the marker, the sidecar, PRE-COMMIT
    snapshots, and ``.chartreux-plan-tmp-*`` temps) never includes the episode's
    transition fence or any provenance-named file, so an ever-allocated episode
    directory is kept — the declared accumulation behavior. Returns ``True``
    only when the directory itself went away; a claim in flight refuses loudly
    and a guarded ``rmdir`` never deletes anything recursive.
    """
    if directory.is_symlink() or not directory.is_dir():
        return False
    if not _is_plan_directory_name(directory.name):
        return False
    return cleanup_episode_workspace(directory, marker_name=PLAN_MARKER_FILENAME)


def _write_plan_marker(directory: Path) -> None:
    """Write the ownership marker; the caller fails allocation if this raises."""
    (directory / PLAN_MARKER_FILENAME).write_text(
        _PLAN_MARKER_CONTENT, encoding="utf-8"
    )


def _discard_created_directory(directory: Path) -> None:
    """Undo a directory this call created, never removing foreign content.

    This is allocation rollback of a directory this call made and never handed
    out, not a fence release: the marker, the sidecar claim, and the fence file
    this call wrote all go, and the directory is removed only when nothing else
    remains.
    """
    for path in (
        directory / PLAN_MARKER_FILENAME,
        sidecar_path(directory),
        fence_path(directory),
    ):
        with suppress(OSError):
            path.unlink()
    with suppress(OSError):
        directory.rmdir()


def _relinquish_marker(directory: Path) -> None:
    """Drop the marker this call wrote, leaving a foreign claim untouched."""
    with suppress(OSError):
        (directory / PLAN_MARKER_FILENAME).unlink()
    with suppress(OSError):
        directory.rmdir()


def _is_plan_directory_name(name: str) -> bool:
    if not name.startswith(PLAN_DIRECTORY_PREFIX):
        return False
    return _SAFE_NAME.fullmatch(name[len(PLAN_DIRECTORY_PREFIX) :]) is not None


def extract_plan_summary(content: str) -> str | None:
    """Return the text under the document's ``## Summary`` heading.

    Returns ``None`` when no such heading exists outside a fenced code block
    and ``""`` when the heading exists but its section holds no text. The
    section ends at the next level-1 or level-2 heading; deeper headings stay
    inside it.
    """
    lines = content.splitlines()
    fence: tuple[str, int] | None = None
    start: int | None = None
    for index, line in enumerate(lines):
        if fence is not None:
            if _fence_closing(line, fence):
                fence = None
            continue
        if (opening := _fence_opening(line)) is not None:
            fence = opening
            continue
        heading = _atx_heading(line)
        if heading is None:
            continue
        level, text = heading
        if start is None:
            if (
                level == _SUMMARY_HEADING_LEVEL
                and text.casefold() == PLAN_SUMMARY_HEADING.casefold()
            ):
                start = index + 1
            continue
        if level <= _SECTION_END_LEVEL:
            return _summarize(lines[start:index])
    return None if start is None else _summarize(lines[start:])


def read_plan_document(path: Path) -> PlanDocument:
    """Read the plan document for acceptance and handoff.

    A usable document is a readable, nonempty file with a nonempty
    ``## Summary`` section; every other outcome raises a
    :class:`PlanDocumentError` subclass so acceptance can report it and start
    nothing.
    """
    if not path.exists():
        raise PlanDocumentMissingError(path)
    if not path.is_file():
        raise PlanDocumentUnreadableError(path)
    try:
        content = read_safe(path, raise_on_error=True).text
    except (OSError, UnicodeDecodeError) as error:
        raise PlanDocumentUnreadableError(
            path, f"could not read plan document: {error}"
        ) from error
    if not content.strip():
        raise PlanDocumentEmptyError(path)
    summary = extract_plan_summary(content)
    if summary is None:
        raise PlanSummaryMissingError(path)
    if not summary:
        raise PlanSummaryEmptyError(path)
    return PlanDocument(path=path, content=content, summary=summary)


def build_planner_seed(workspace: PlanWorkspace, theme: str | None = None) -> str:
    """Orient the planner, then defer design, planning, and writing until asked."""
    theme_line = (
        f"\nSession theme: {theme}. The theme is a heading, not an instruction "
        "or authorization."
        if theme
        else ""
    )
    return (
        f"Plan mode is active; permissions are unchanged.{theme_line}\n\n"
        "Orient yourself about the state of the repository first: read the "
        "current branch, the recent releases, and the existing workspace tasks. "
        "Then present a brief orientation to the user and stand by for the plan "
        "agenda. Do not infer or invent a task, do not propose agenda items or "
        "work on your own — the user drives — and do not begin design, "
        "planning, or implementation until the user supplies the agenda. Never "
        "start implementation yourself: the accepted plan is executed by a "
        "separate session.\n\n"
        "The plan document is written only after its design and the plan itself "
        "have been accepted by the user, as Markdown to:\n"
        f"{workspace.document_path}\n\n"
        f'Include a "## {PLAN_SUMMARY_HEADING}" section: its text is the brief '
        "handed to the execution session."
    )


def build_plan_mode_compaction_notice(
    plan_path: str,
    theme: str | None = None,
    *,
    episode: EpisodeObservation | None = None,
    episode_failure: str | None = None,
) -> str:
    """The one-line notice re-injected after a compaction rewrote the history.

    The same text is the ``/plan status`` line. The theme clause follows the
    seed's iff-theme discipline: a themeless mode never renders
    "Theme: None.".

    ``episode`` is the record's sidecar-first observation (design §4.5). The
    record's state is a mirror, so the re-injected binding follows the SIDECAR:
    a draft or accepting episode re-injects its live status, while a terminal,
    possibly-executed, or unresolvable episode is re-injected as history that is
    NOT active planning — a compaction can never re-arm a terminal episode.
    ``episode_failure`` reports an untrusted arbiter read: the binding cannot be
    asserted, so the mode is not re-injected as active.
    """
    if episode_failure is not None:
        return (
            f"Plan episode state could not be validated ({episode_failure}); "
            "plan mode is not re-injected. Rebind to a fresh episode or start "
            f"new. Plan document: {plan_path}."
        )
    if episode is not None and episode.terminal:
        return (
            f"Plan episode {episode.episode_id or '(unidentified)'} is "
            f"{episode.state or 'terminal'} (terminal): plan mode is not active "
            "in this session. Start a new episode to plan again. Plan document: "
            f"{plan_path}."
        )
    if episode is not None and episode.possibly_executed:
        return (
            f"Plan episode {episode.episode_id or '(unidentified)'} was already "
            "executed by session "
            f"{episode.executor_session_id or '(unrecorded)'}: plan mode is not "
            f"active in this session. Plan document: {plan_path}."
        )
    if episode is not None and episode.row == RECOVERY_ROW_UNRESOLVABLE:
        return (
            f"Plan episode {episode.episode_id or '(unidentified)'} is "
            "unresolvable (no sidecar and no executor session): rebind to a "
            "fresh episode or start new; plan mode is not active in this "
            f"session. Plan document: {plan_path}."
        )
    notice = f"Plan mode is active. Plan document: {plan_path}."
    if episode is not None and episode.episode_id is not None:
        notice = f"{notice} Plan episode {episode.episode_id}: {episode.state}."
    if theme:
        return f"{notice} Theme: {theme}."
    return notice


def build_executor_seed(document: PlanDocument) -> str:
    """The handoff brief: the displayed summary plus the absolute plan path."""
    return (
        f"Execute the plan at {document.path}.\n\n"
        f"Planner summary:\n{document.summary}\n\n"
        "Read the plan document at that path before executing."
    )


def _atx_heading(line: str) -> tuple[int, str] | None:
    match = _ATX_HEADING.fullmatch(line)
    if match is None:
        return None
    return len(match.group(1)), (match.group(2) or "").strip()


def _fence_opening(line: str) -> tuple[str, int] | None:
    match = _FENCE_OPEN.fullmatch(line)
    if match is None:
        return None
    marker, info = match.group(1), match.group(2)
    if marker[0] == "`" and "`" in info:
        return None
    return marker[0], len(marker)


def _fence_closing(line: str, fence: tuple[str, int]) -> bool:
    match = _FENCE_CLOSE.fullmatch(line)
    if match is None:
        return False
    marker = match.group(1)
    return marker[0] == fence[0] and len(marker) >= fence[1]


def _summarize(lines: list[str]) -> str:
    return "\n".join(lines).strip()
