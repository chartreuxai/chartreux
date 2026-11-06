from __future__ import annotations

import os
from pathlib import Path
import stat

import pytest

from chartreux.core.config.models import PlanExecutorConfig, PlanSectionConfig
from chartreux.core.plan_mode import (
    PLAN_DIRECTORY_PREFIX,
    PLAN_DOCUMENT_FILENAME,
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
    build_planner_seed,
    default_workspaces_root,
    derive_project_name,
    derive_session_suffix,
    executor_selection,
    extract_plan_summary,
    planner_selection,
    read_plan_document,
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


def test_executor_seed_carries_the_summary_and_the_plan_path(tmp_path: Path) -> None:
    document = tmp_path / PLAN_DOCUMENT_FILENAME
    document.write_text("## Summary\nShip it.\n\n## Steps\n1. One\n", encoding="utf-8")
    plan = read_plan_document(document)

    seed = build_executor_seed(plan)

    assert str(document) in seed
    assert "Ship it." in seed
    assert "before executing" in seed
