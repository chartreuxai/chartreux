"""Workspace prompt, branch, trust, git worktree, and checkout wire models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from chartreux.app_server._model import ProtocolModel
from chartreux.app_server.models import (
    ContentBlock,
    PreparedPrompt,
    WorkspaceTrustDecision,
    WorkspaceTrustDetails,
    WorkspaceTrustStatus,
)


class WorkspacePromptPrepareParams(ProtocolModel):
    session_id: str
    message: str
    title_content: list[ContentBlock] | None = None


class WorkspacePromptPrepareResponse(ProtocolModel):
    prompt: PreparedPrompt


class WorkspaceBranchReadParams(ProtocolModel):
    session_id: str
    cwd: str = Field(min_length=1)


class WorkspaceBranchReadResponse(ProtocolModel):
    session_id: str
    cwd: str
    branch: str | None = None
    status: Literal["branch", "detached", "not_repository", "unknown"]


class WorkspaceTrustStatusParams(ProtocolModel):
    cwd: str | None = None


class WorkspaceTrustStatusResponse(ProtocolModel):
    status: WorkspaceTrustStatus
    details: WorkspaceTrustDetails | None = None


class WorkspaceWorktreeListParams(ProtocolModel):
    cwd: str = Field(min_length=1)
    # Off by default because this listing sits on the read path: it resolves
    # the checkout behind every session read and enumerates a project's
    # directories for every session list. The details cost a merge base and a
    # diff per branch plus a second repository open, which only a caller that
    # renders them should pay.
    include_details: bool = False


class WorkspaceGitBranchChanges(ProtocolModel):
    additions: int
    deletions: int


class WorkspaceLinkedWorktree(ProtocolModel):
    name: str
    branch: str
    cwd: str
    root: str
    repo_root: str
    # Absent unless asked for, and null when there is no base to measure
    # against, which is not the same as a branch that has changed nothing.
    branch_changes: WorkspaceGitBranchChanges | None = None


class WorkspaceWorktreeListResponse(ProtocolModel):
    worktrees: list[WorkspaceLinkedWorktree]
    # The branch the main checkout is on. Absent unless details were asked for,
    # and null for a detached one. The worktree entries never name it: this
    # listing reports the linked worktrees, and the main checkout is not one.
    repository_branch: str | None = None
    # Where the position this listing was taken from sits in the main checkout,
    # under the same checks the worktree entries pass. Null when it does not
    # sit there at all -- a subdirectory that exists only on a feature branch
    # has no counterpart. A caller offering the main checkout as a destination
    # must take this rather than joining the root itself, because a path this
    # omits is one a move would refuse.
    repository_cwd: str | None = None


# No session_id on the wire: the caller is deleting a session that has already
# closed and dropped its holder, and accepting one would let a client name an
# arbitrary holder file to unlink.
class WorkspaceWorktreeRemoveParams(ProtocolModel):
    cwd: str = Field(min_length=1)


# Spelled out here rather than imported from chartreux.core.git.worktree: the protocol
# is the wire contract and must not pull core into the app-server clients.
type WorktreeRemoveOutcome = Literal[
    "removed",
    "kept_dirty",
    "kept_in_use",
    "kept_unmanaged",
    # Distinct from kept_unmanaged: the worktree is Chartreux's and the removal
    # itself failed. Collapsing the two would report a failure as "not ours".
    "kept_error",
    "not_found",
]


class WorkspaceWorktreeRemoveResponse(ProtocolModel):
    # A kept worktree is a normal outcome the caller has to render, not a fault,
    # so every case answers with a result rather than a JSON-RPC error.
    outcome: WorktreeRemoveOutcome
    root: str | None = None
    branch: str | None = None
    branch_deleted: bool = False
    reasons: list[str] = Field(default_factory=list)


class WorkspaceGitCheckoutsParams(ProtocolModel):
    # Every repository the project links, asked for together, because which one
    # holds the session cannot be decided from any single one. A managed
    # worktree lives outside the repository it belongs to, and a repository
    # linked inside another would otherwise let both claim the session.
    repo_local_paths: list[str]
    # Absent for a session with no working directory, which on a cloud host is
    # every session.
    session_cwd: str | None = None


class WorkspaceGitCheckout(ProtocolModel):
    repo_local_path: str
    # False when the repository could not be read; `message` says why and the
    # rest is absent. Carried rather than raised so one unreadable repository
    # does not cost the answer for the others.
    ok: bool
    # The repository the session is standing in. At most one is.
    is_primary: bool = False
    repo_url: str | None = None
    root: str | None = None
    # Absent when the session sits in the repository's own checkout rather than
    # in one of its worktrees.
    worktree: str | None = None
    branch: str | None = None
    base_branch: str | None = None
    message: str | None = None


class WorkspaceGitCheckoutsResponse(ProtocolModel):
    checkouts: list[WorkspaceGitCheckout] = Field(default_factory=list)


class WorkspaceTrustDecisionParams(ProtocolModel):
    decision: WorkspaceTrustDecision
    cwd: str | None = None
    session_id: str | None = None


class WorkspaceUntrustedConfigParams(ProtocolModel):
    cwd: str | None = None


class WorkspaceUntrustedConfigResponse(ProtocolModel):
    dirs: list[str] = Field(default_factory=list)
    settings_path: str = ""
