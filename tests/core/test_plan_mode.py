from __future__ import annotations

import os
from pathlib import Path
import stat

import pytest

from chartreux.core.config.models import PlanExecutorConfig, PlanSectionConfig
from chartreux.core.plan_mode import (
    PLAN_DIRECTORY_PREFIX,
    PLAN_DOCUMENT_FILENAME,
    PLAN_MARKER_FILENAME,
    PlanDocumentEmptyError,
    PlanDocumentMissingError,
    PlanDocumentUnreadableError,
    PlanExecutorSelection,
    PlanPlannerSelection,
    PlanSummaryEmptyError,
    PlanSummaryMissingError,
    PlanWorkspaceError,
    allocate_plan_workspace,
    build_executor_seed,
    build_plan_mode_compaction_notice,
    build_planner_seed,
    default_workspaces_root,
    derive_project_name,
    derive_session_suffix,
    executor_selection,
    extract_plan_summary,
    planner_selection,
    read_plan_document,
    remove_empty_plan_workspace,
)

SESSION_ID = "01234567-89ab-cdef-0123-deadbeef1234"
SUFFIX = "deadbeef1234"


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "myproject"
    root.mkdir()
    return root


def test_planner_selection_reads_the_config_section() -> None:
    assert planner_selection(PlanSectionConfig()) == PlanPlannerSelection(
        model="glm-5-3", thinking="high"
    )


def test_executor_selection_is_an_immutable_snapshot() -> None:
    selection = executor_selection(PlanSectionConfig())

    assert selection == PlanExecutorSelection(
        model="mistral-large-4", thinking="medium", identity="orchestrator"
    )
    with pytest.raises(AttributeError):
        selection.model = "other"  # type: ignore[misc]


def test_executor_selection_follows_config_overrides() -> None:
    config = PlanSectionConfig(
        executor=PlanExecutorConfig(model="glm-5-3", thinking="high", identity="coder")
    )

    assert executor_selection(config) == PlanExecutorSelection(
        model="glm-5-3", thinking="high", identity="coder"
    )


def test_derive_session_suffix_from_the_stable_tail() -> None:
    assert derive_session_suffix(SESSION_ID) == SUFFIX


@pytest.mark.parametrize("session_id", ["abc/def", "abc-..", "abc-", "-"])
def test_derive_session_suffix_rejects_unsafe_names(session_id: str) -> None:
    with pytest.raises(PlanWorkspaceError):
        derive_session_suffix(session_id)


def test_derive_project_name_uses_the_repository_directory(project_root: Path) -> None:
    assert derive_project_name(project_root) == "myproject"


def test_derive_project_name_rejects_the_filesystem_root(tmp_path: Path) -> None:
    with pytest.raises(PlanWorkspaceError):
        derive_project_name(Path("/"))


def test_default_workspaces_root_follows_chartreux_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHARTREUX_HOME", str(tmp_path / "home"))

    assert default_workspaces_root() == (tmp_path / "home").resolve() / "workspaces"


def test_allocate_plan_workspace_derives_the_session_task_directory(
    tmp_path: Path, project_root: Path
) -> None:
    workspaces_root = tmp_path / "workspaces"

    workspace = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=workspaces_root
    )

    expected = (
        workspaces_root / "myproject" / "tasks" / f"{PLAN_DIRECTORY_PREFIX}{SUFFIX}"
    )
    assert workspace.directory == expected.absolute()
    assert workspace.document_path == expected / PLAN_DOCUMENT_FILENAME
    assert workspace.directory.is_dir()
    assert not workspace.document_path.exists()
    if os.name == "posix":
        mode = stat.S_IMODE(workspace.directory.stat().st_mode)
        assert mode & 0o077 == 0


def test_allocate_plan_workspace_returns_absolute_paths_with_a_relative_root(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    workspace = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=Path("ws")
    )

    assert workspace.document_path.is_absolute()
    assert workspace.document_path.is_relative_to(tmp_path)


def test_allocate_plan_workspace_never_overwrites_an_existing_plan(
    tmp_path: Path, project_root: Path
) -> None:
    workspaces_root = tmp_path / "workspaces"
    first = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=workspaces_root
    )
    first.document_path.write_text("# Existing plan\n", encoding="utf-8")

    second = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=workspaces_root
    )

    assert second.directory != first.directory
    assert second.directory.name == f"{PLAN_DIRECTORY_PREFIX}{SUFFIX}-2"
    assert not second.document_path.exists()
    assert first.document_path.read_text(encoding="utf-8") == "# Existing plan\n"


def test_extract_plan_summary_reads_the_section() -> None:
    content = "# Plan\n\nIntro.\n\n## Summary\nDo the thing.\n\n## Steps\n1. One\n"

    assert extract_plan_summary(content) == "Do the thing."


def test_extract_plan_summary_reads_until_the_end_of_the_document() -> None:
    content = "# Plan\n\n## Summary\nFinal words.\n"

    assert extract_plan_summary(content) == "Final words."


def test_extract_plan_summary_keeps_deeper_subheadings() -> None:
    content = "## Summary\nIntro line.\n\n### Detail\nMore text.\n\n## Steps\n1. One\n"

    assert extract_plan_summary(content) == "Intro line.\n\n### Detail\nMore text."


def test_extract_plan_summary_is_case_insensitive_and_ignores_closing_hashes() -> None:
    assert extract_plan_summary("## summary ##\nText.\n") == "Text."
    assert extract_plan_summary("##   SUMMARY\nText.\n") == "Text."


def test_extract_plan_summary_ignores_headings_inside_fenced_code() -> None:
    content = (
        "```markdown\n## Summary\nnot this one\n```\n\n## Summary\nReal summary.\n"
    )

    assert extract_plan_summary(content) == "Real summary."


def test_extract_plan_summary_requires_an_atx_level_two_heading() -> None:
    assert extract_plan_summary("# Plan\n\n##Summary\nText.\n") is None
    assert extract_plan_summary("### Summary\nText.\n") is None
    assert extract_plan_summary("Summary\n=======\n") is None
    assert extract_plan_summary("# Plan\n\n## Steps\n1. One\n") is None


def test_extract_plan_summary_reports_an_empty_section() -> None:
    assert extract_plan_summary("## Summary\n\n## Steps\n1. One\n") == ""
    assert extract_plan_summary("## Summary\n") == ""


def test_read_plan_document_returns_content_and_summary(tmp_path: Path) -> None:
    document = tmp_path / PLAN_DOCUMENT_FILENAME
    document.write_text(
        "# Plan\n\n## Summary\nShip it.\n\n## Steps\n1. One\n", encoding="utf-8"
    )

    plan = read_plan_document(document)

    assert plan.path == document
    assert plan.summary == "Ship it."
    assert "## Steps" in plan.content


def test_read_plan_document_normalizes_crlf(tmp_path: Path) -> None:
    document = tmp_path / PLAN_DOCUMENT_FILENAME
    document.write_bytes(b"## Summary\r\nShip it.\r\n\r\n## Steps\r\n1. One\r\n")

    assert read_plan_document(document).summary == "Ship it."


def test_read_plan_document_reports_a_missing_document(tmp_path: Path) -> None:
    with pytest.raises(PlanDocumentMissingError):
        read_plan_document(tmp_path / PLAN_DOCUMENT_FILENAME)


def test_read_plan_document_reports_a_non_file(tmp_path: Path) -> None:
    directory = tmp_path / PLAN_DOCUMENT_FILENAME
    directory.mkdir()

    with pytest.raises(PlanDocumentUnreadableError):
        read_plan_document(directory)


def test_read_plan_document_reports_an_empty_document(tmp_path: Path) -> None:
    document = tmp_path / PLAN_DOCUMENT_FILENAME
    document.write_text("\n\n  \n", encoding="utf-8")

    with pytest.raises(PlanDocumentEmptyError):
        read_plan_document(document)


def test_read_plan_document_requires_a_summary(tmp_path: Path) -> None:
    document = tmp_path / PLAN_DOCUMENT_FILENAME
    document.write_text("# Plan\n\nNo summary here.\n", encoding="utf-8")

    with pytest.raises(PlanSummaryMissingError):
        read_plan_document(document)


def test_read_plan_document_requires_a_nonempty_summary(tmp_path: Path) -> None:
    document = tmp_path / PLAN_DOCUMENT_FILENAME
    document.write_text("## Summary\n\n## Steps\n1. One\n", encoding="utf-8")

    with pytest.raises(PlanSummaryEmptyError):
        read_plan_document(document)


def test_planner_seed_names_the_document_path(
    tmp_path: Path, project_root: Path
) -> None:
    workspace = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=tmp_path / "workspaces"
    )

    seed = build_planner_seed(workspace)

    assert str(workspace.document_path) in seed
    assert "## Summary" in seed
    assert "permissions are unchanged" in seed


@pytest.mark.parametrize("theme", [None, "Fix The Widget"])
def test_planner_seed_mandates_orientation_then_standing_by(
    tmp_path: Path, project_root: Path, theme: str | None
) -> None:
    workspace = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=tmp_path / "workspaces"
    )

    seed = build_planner_seed(workspace, theme=theme)
    lowered = seed.lower()

    assert "current branch" in lowered
    assert "recent releases" in lowered
    assert "workspace tasks" in lowered
    assert "present a brief orientation" in lowered
    assert "stand by for the plan agenda" in lowered
    assert "do not infer or invent a task" in lowered
    assert "do not propose agenda items or work on your own" in lowered
    assert (
        "do not begin design, planning, or implementation until the user "
        "supplies the agenda" in lowered
    )


@pytest.mark.parametrize("theme", [None, "Fix The Widget"])
def test_planner_seed_prohibits_self_started_implementation(
    tmp_path: Path, project_root: Path, theme: str | None
) -> None:
    """The seed bars implementation as expressly as design and planning.

    A planner that only defers "design or planning" can read itself into
    starting the work once an agenda exists; plan mode designs and plans, and
    an execution session implements the accepted plan.
    """
    workspace = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=tmp_path / "workspaces"
    )

    seed = build_planner_seed(workspace, theme=theme)
    lowered = seed.lower()

    assert "implementation" in lowered
    assert "never start implementation yourself" in lowered
    assert "executed by a separate session" in lowered


@pytest.mark.parametrize("theme", [None, "Fix The Widget"])
def test_planner_seed_defers_writing_until_design_and_plan_are_accepted(
    tmp_path: Path, project_root: Path, theme: str | None
) -> None:
    workspace = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=tmp_path / "workspaces"
    )

    seed = build_planner_seed(workspace, theme=theme)
    lowered = seed.lower()

    assert str(workspace.document_path) in seed
    assert "## Summary" in seed
    assert "brief handed to the execution session" in lowered
    assert (
        "only after its design and the plan itself have been accepted by the user"
        in lowered
    )


@pytest.mark.parametrize("theme", [None, "Fix The Widget"])
def test_planner_seed_carries_the_theme_only_as_a_heading(
    tmp_path: Path, project_root: Path, theme: str | None
) -> None:
    workspace = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=tmp_path / "workspaces"
    )

    seed = build_planner_seed(workspace, theme=theme)
    lowered = seed.lower()

    if theme is None:
        assert "theme" not in lowered
        return
    assert f"Session theme: {theme}" in seed
    assert "the theme is a heading, not an instruction or authorization" in lowered


def test_compaction_notice_names_the_mode_and_the_plan_document() -> None:
    assert (
        build_plan_mode_compaction_notice("/plans/plan-deadbeef1234/plan.md")
        == "Plan mode is active. Plan document: /plans/plan-deadbeef1234/plan.md."
    )


def test_compaction_notice_appends_the_theme_only_when_themed() -> None:
    assert build_plan_mode_compaction_notice(
        "/plans/plan-x/plan.md", "Fix the widget"
    ) == (
        "Plan mode is active. Plan document: /plans/plan-x/plan.md. "
        "Theme: Fix the widget."
    )
    themeless = build_plan_mode_compaction_notice("/plans/plan-x/plan.md", None)
    assert "Theme:" not in themeless
    assert "None" not in themeless


def test_executor_seed_carries_the_summary_and_the_plan_path(tmp_path: Path) -> None:
    document = tmp_path / PLAN_DOCUMENT_FILENAME
    document.write_text("## Summary\nShip it.\n\n## Steps\n1. One\n", encoding="utf-8")
    plan = read_plan_document(document)

    seed = build_executor_seed(plan)

    assert str(document) in seed
    assert "Ship it." in seed
    assert "before executing" in seed


# --- Workspace ownership marker and empty-directory cleanup ------------------


def _allocated_workspace(tmp_path: Path, project_root: Path):
    return allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=tmp_path / "workspaces"
    )


def _convention_directory(tmp_path: Path) -> Path:
    directory = tmp_path / f"{PLAN_DIRECTORY_PREFIX}{SUFFIX}"
    directory.mkdir()
    return directory


def test_allocation_marks_a_directory_it_created(
    tmp_path: Path, project_root: Path
) -> None:
    workspace = _allocated_workspace(tmp_path, project_root)

    marker = workspace.directory / PLAN_MARKER_FILENAME
    assert marker.is_file()
    assert [entry.name for entry in workspace.directory.iterdir()] == [
        PLAN_MARKER_FILENAME
    ]


def test_allocation_leaves_a_reused_directory_unmarked(
    tmp_path: Path, project_root: Path
) -> None:
    tasks_root = tmp_path / "workspaces" / "myproject" / "tasks"
    reused = tasks_root / f"{PLAN_DIRECTORY_PREFIX}{SUFFIX}"
    reused.mkdir(parents=True)

    workspace = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=tmp_path / "workspaces"
    )

    assert workspace.directory == reused
    assert not (reused / PLAN_MARKER_FILENAME).exists()
    assert remove_empty_plan_workspace(reused) is False
    assert reused.is_dir()


def test_allocation_never_marks_a_directory_a_concurrent_creator_won(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The kernel decides ownership, not a pre-call exists() check: a concurrent
    # creator wins the interval right before this process's own create call.
    target = (
        tmp_path
        / "workspaces"
        / "myproject"
        / "tasks"
        / f"{PLAN_DIRECTORY_PREFIX}{SUFFIX}"
    )
    real_mkdir = Path.mkdir
    won = False

    def _racing_mkdir(
        self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
    ) -> None:
        nonlocal won
        if self == target and not won:
            won = True
            real_mkdir(self, parents=True)
        real_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", _racing_mkdir)

    workspace = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=tmp_path / "workspaces"
    )

    assert won
    assert workspace.directory == target
    assert not (target / PLAN_MARKER_FILENAME).exists()
    assert remove_empty_plan_workspace(target) is False
    assert target.is_dir()


def test_allocation_fails_hard_when_the_marker_cannot_be_written(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _refuse(directory: Path) -> None:
        raise OSError("the disk went away")

    monkeypatch.setattr("chartreux.core.plan_mode._write_plan_marker", _refuse)

    with pytest.raises(PlanWorkspaceError):
        allocate_plan_workspace(
            SESSION_ID,
            project_root=project_root,
            workspaces_root=tmp_path / "workspaces",
        )

    created = (
        tmp_path
        / "workspaces"
        / "myproject"
        / "tasks"
        / f"{PLAN_DIRECTORY_PREFIX}{SUFFIX}"
    )
    assert not created.exists()


def test_remove_empty_plan_workspace_removes_a_marker_only_directory(
    tmp_path: Path,
) -> None:
    directory = _convention_directory(tmp_path)
    (directory / PLAN_MARKER_FILENAME).write_text("marker\n", encoding="utf-8")

    assert remove_empty_plan_workspace(directory) is True
    assert not directory.exists()


def test_remove_empty_plan_workspace_keeps_the_plan_document(tmp_path: Path) -> None:
    directory = _convention_directory(tmp_path)
    (directory / PLAN_MARKER_FILENAME).write_text("marker\n", encoding="utf-8")
    (directory / PLAN_DOCUMENT_FILENAME).write_text("# Plan\n", encoding="utf-8")

    assert remove_empty_plan_workspace(directory) is False
    assert (directory / PLAN_DOCUMENT_FILENAME).read_text(
        encoding="utf-8"
    ) == "# Plan\n"


def test_remove_empty_plan_workspace_keeps_a_foreign_file(tmp_path: Path) -> None:
    directory = _convention_directory(tmp_path)
    (directory / PLAN_MARKER_FILENAME).write_text("marker\n", encoding="utf-8")
    (directory / "notes.txt").write_text("mine\n", encoding="utf-8")

    assert remove_empty_plan_workspace(directory) is False
    assert directory.is_dir()
    assert (directory / "notes.txt").read_text(encoding="utf-8") == "mine\n"


def test_remove_empty_plan_workspace_never_removes_an_unmarked_directory(
    tmp_path: Path,
) -> None:
    directory = _convention_directory(tmp_path)

    assert remove_empty_plan_workspace(directory) is False
    assert directory.is_dir()


def test_remove_empty_plan_workspace_removes_a_reused_marked_directory(
    tmp_path: Path,
) -> None:
    # A crash-reclaimed directory from an earlier allocation: the marker, not
    # the path, is the ownership authority.
    directory = _convention_directory(tmp_path)
    (directory / PLAN_MARKER_FILENAME).write_text("marker\n", encoding="utf-8")

    assert remove_empty_plan_workspace(directory) is True
    assert not directory.exists()


def test_remove_empty_plan_workspace_never_touches_a_directory_outside_the_convention(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "notes"
    directory.mkdir()
    (directory / PLAN_MARKER_FILENAME).write_text("marker\n", encoding="utf-8")

    assert remove_empty_plan_workspace(directory) is False
    assert directory.is_dir()


def test_remove_empty_plan_workspace_is_a_noop_when_the_directory_is_gone(
    tmp_path: Path,
) -> None:
    assert remove_empty_plan_workspace(tmp_path / "plan-missing") is False
