from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel

from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.tools.utils import resolve_tool_path
from chartreux.questions import (
    QuestionChoice,
    UserQuestion,
    UserQuestionRequest,
    UserQuestionResult,
)

ROOT_GRANT_TOOLS = frozenset({"read_file", "write_file", "edit", "grep", "read_image"})
ROOT_GRANT_TARGET_ARG = {
    "read_file": "file_path",
    "write_file": "file_path",
    "edit": "file_path",
    "read_image": "file_path",
    "grep": "path",
}
ROOT_GRANT_SESSION_LABEL = "Allow this session"
ROOT_GRANT_PROJECT_LABEL = "Always for this project"
ROOT_GRANT_DENY_LABEL = "Deny"
ROOT_GRANT_DECLINED_FEEDBACK = "user declined; do not retry this path"

ROOT_GRANT_SESSION = "session"
ROOT_GRANT_PROJECT = "project"
ROOT_GRANT_DENY = "deny"


class SessionRootGrants:
    """Bare canonical roots and deny-memory owned by one agent loop."""

    def __init__(self) -> None:
        self.roots: set[Path] = set()
        self.denied_roots: set[Path] = set()
        self.pending: dict[Path, asyncio.Future[str]] = {}

    def grant(self, root: Path) -> bool:
        canonical = root.resolve()
        self.denied_roots.discard(canonical)
        before = len(self.roots)
        self.roots.add(canonical)
        return len(self.roots) != before

    def deny(self, root: Path) -> None:
        self.denied_roots.add(root.resolve())


def root_grant_target(tool_name: str, args: BaseModel, cwd: Path) -> Path | None:
    """Resolve the exact target a file tool's permission check denied."""
    raw = getattr(args, ROOT_GRANT_TARGET_ARG[tool_name], None)
    if not isinstance(raw, str) or not raw.strip():
        return None
    return resolve_tool_path(raw, cwd)


def propose_root_grant(target: Path) -> Path | None:
    """Narrowest existing directory covering *target*.

    An existing directory target proposes itself; anything else proposes its
    nearest existing ancestor, never a wider parent of an existing directory.
    """
    if target.is_dir():
        return target
    for ancestor in target.parents:
        if ancestor.is_dir():
            return ancestor
    return None


def root_grant_request(root: Path) -> UserQuestionRequest:
    """Build the mid-turn question asking the user to grant *root*."""
    question = (
        f"Grant {root} to this session? It allows reading and writing files "
        f"under {root}, including with shell commands."
    )
    home = Path.home()
    if root == home or home.is_relative_to(root):
        question += (
            " This root is your home directory or wider, so it exposes most "
            "of your filesystem."
        )
    return UserQuestionRequest(
        questions=[
            UserQuestion(
                question=question,
                header="Grant access",
                options=[
                    QuestionChoice(
                        label=ROOT_GRANT_SESSION_LABEL,
                        description=(
                            "Read and write under this path for this session only"
                        ),
                    ),
                    QuestionChoice(
                        label=ROOT_GRANT_PROJECT_LABEL,
                        description=(
                            "Grant for this session and save it to your user "
                            "config for this project"
                        ),
                    ),
                    QuestionChoice(
                        label=ROOT_GRANT_DENY_LABEL,
                        description="Keep the path unauthorized and skip this call",
                    ),
                ],
                hide_other=True,
            )
        ],
        footer_note=(
            "Session grants stay in memory; only 'Always for this project' "
            "writes to your user config."
        ),
    )


def parse_root_grant_result(result: BaseModel) -> str:
    """Map a user answer to a grant decision; anything else fails closed."""
    if isinstance(result, UserQuestionResult) and not result.cancelled:
        for answer in result.answers:
            if answer.answer == ROOT_GRANT_SESSION_LABEL:
                return ROOT_GRANT_SESSION
            if answer.answer == ROOT_GRANT_PROJECT_LABEL:
                return ROOT_GRANT_PROJECT
    return ROOT_GRANT_DENY


def root_grant_save_note(root: Path, project: Path, result: ConfigSaveResult) -> str:
    """Report a project-grant save apart from the session grant it follows.

    The session grant is already applied when this note is built; the note
    carries only the persistence outcome, honestly, including the unchanged
    application semantics of the save path.
    """
    if result.error == "cancelled":
        return (
            f"Root grant for {root} applies to this session; the save to your "
            "user config was cancelled before it could be confirmed, so treat "
            "the grant as session-only."
        )
    if result.persistence == "durability_uncertain":
        return (
            f"Root grant for {root} applies to this session; the save to your "
            "user config could not be confirmed, so treat the grant as "
            "session-only."
        )
    if result.persistence == "saved" and result.error is None:
        return (
            f"Root grant for {root} applies to this session and is saved in "
            f"your user config for project {project}. The running session's "
            "config is unchanged; future sessions start with the saved root."
        )
    if result.error == "conflict":
        return (
            f"Root grant for {root} applies to this session only; saving it to "
            "your user config was rejected because the config changed since it "
            "was read. You can retry 'Always for this project' to save it."
        )
    if result.error == "write":
        return (
            f"Root grant for {root} applies to this session only; saving it to "
            "your user config failed."
        )
    return (
        f"Root grant for {root} applies to this session only; saving it to "
        "your user config was rejected."
    )
