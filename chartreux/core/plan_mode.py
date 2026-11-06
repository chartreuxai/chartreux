"""Plan-mode document helpers: selections, workspace allocation, reads, seeds.

``/plan`` keeps one authoritative Markdown document per planning session under
the project's workspace ``tasks/`` directory. The planner writes it; the
execution session is seeded with its ``## Summary`` text and its absolute path.
This module owns the path derivation, the non-destructive allocation, and the
document reads shared by the TUI and the app server, plus the immutable
selection snapshots the consent modal displays and the handoff applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from chartreux.config_values import ThinkingLevel
from chartreux.core.config.models import PlanSectionConfig
from chartreux.core.paths import CHARTREUX_HOME
from chartreux.core.session.session_id import extract_suffix
from chartreux.utils.io import read_safe
from chartreux.utils.private_paths import ensure_private_directory

PLAN_DIRECTORY_PREFIX = "plan-"
PLAN_DOCUMENT_FILENAME = "plan.md"
PLAN_SUMMARY_HEADING = "Summary"

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
    """The allocated task directory and the document path inside it."""

    directory: Path
    document_path: Path


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
    """Allocate the session-derived plan workspace, never overwriting a plan.

    Uses ``<workspaces_root>/<project>/tasks/plan-<suffix>/plan.md``; when that
    document already exists, the first free numbered alternative
    (``plan-<suffix>-2``, ...) is allocated instead. The task directory is
    created owner-only; the document itself is left for the planner to write.
    """
    suffix = derive_session_suffix(session_id)
    project_name = derive_project_name(project_root)
    root = workspaces_root if workspaces_root is not None else default_workspaces_root()
    tasks_root = (root / project_name / "tasks").absolute()
    base = f"{PLAN_DIRECTORY_PREFIX}{suffix}"
    for attempt in range(1, _MAX_NAME_ATTEMPTS + 1):
        directory = tasks_root / (base if attempt == 1 else f"{base}-{attempt}")
        document_path = directory / PLAN_DOCUMENT_FILENAME
        if document_path.exists():
            continue
        ensure_private_directory(directory)
        if not directory.is_dir():
            raise PlanWorkspaceError(
                f"cannot create plan workspace directory: {directory}"
            )
        if document_path.exists():
            continue
        return PlanWorkspace(directory=directory, document_path=document_path)
    raise PlanWorkspaceError(
        f"no free plan workspace name for session {session_id!r} under {tasks_root}"
    )


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


def build_planner_seed(workspace: PlanWorkspace) -> str:
    """The visible entry message telling the planner where to write the plan."""
    return (
        "Plan mode is active. Explore and research freely; permissions are "
        "unchanged. Write the plan as Markdown to:\n"
        f"{workspace.document_path}\n\n"
        f'Include a "## {PLAN_SUMMARY_HEADING}" section: its text is the brief '
        "handed to the execution session."
    )


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
