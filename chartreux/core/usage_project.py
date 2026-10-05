from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

from chartreux.core.git.errors import GitError
from chartreux.core.git.repo import GitRepo


def resolve_project_key(root_workspace: str | Path) -> str:
    """Resolve the root's local identity once, then inherit it in all children.

    Git checkouts use their canonical common Git directory, so linked worktrees
    share a key while separate clones do not. No remote URL is inspected. If Git
    is unavailable, or the directory is missing or not a repository, use the
    canonical workspace path instead. This fallback may change when Git becomes
    available; callers must retain the resolved key for the root's lifetime.

    This performs blocking filesystem/Git work. Call from a worker thread or use
    ``resolve_project_key_async`` on an event loop, never from a child's cwd.
    Keys are namespaced 128-bit SHA-256 prefixes, not Python's randomized hashes.
    """
    identity = Path(root_workspace).resolve()
    namespace = "path"
    try:
        with GitRepo.open(identity) as repo:
            identity = repo.paths.common_git_dir.resolve()
            namespace = "git"
    except (GitError, OSError):
        pass
    digest = hashlib.sha256(os.fsencode(identity)).hexdigest()[:32]
    return f"{namespace}-{digest}"


async def resolve_project_key_async(root_workspace: str | Path) -> str:
    """Resolve root attribution without blocking the event loop."""
    return await asyncio.to_thread(resolve_project_key, root_workspace)
