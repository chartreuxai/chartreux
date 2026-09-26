from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import fnmatch
import ntpath
import os
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
import posixpath

from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.scratchpad import is_scratchpad_path
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.workspace import Workspace
from chartreux.utils.paths import is_foreign_windows_path

# A model-supplied path argument, normalized once when the tool call is validated.
ToolPath = str

DEFAULT_SENSITIVE_PATTERNS: list[str] = [
    "**/.env",
    "**/.env.*",
    "**/.env~",
    "**/.envrc",
    "**/.envrc.*",
    "**/.envrc~",
]


def matches_sensitive_pattern(resolved_path: str, patterns: list[str]) -> bool:
    """Return True if a resolved path matches any sensitive glob, case-insensitively."""
    lowered = PurePath(resolved_path.lower())
    return any(lowered.match(pattern.lower()) for pattern in patterns)


_active_file_display_harness: ContextVar[HarnessFilesManager | None] = ContextVar(
    "active_file_display_harness", default=None
)


@contextmanager
def file_display_harness(harness_files: HarnessFilesManager | None) -> Iterator[None]:
    if harness_files is None:
        yield
        return

    token = _active_file_display_harness.set(harness_files)
    try:
        yield
    finally:
        _active_file_display_harness.reset(token)


def _make_absolute(path_str: str, cwd: Path) -> Path:
    path = Path(path_str).expanduser()
    if is_foreign_windows_path(path_str):
        return path
    if path.is_absolute():
        return path
    return cwd / path


def resolve_tool_path(raw: str | None, cwd: Path) -> Path:
    """Resolve a model-supplied path against the tool's working directory."""
    if not raw:
        return cwd
    path = _make_absolute(raw, cwd)
    # Preserve foreign path identity until permission checks can reject it.
    # POSIX resolve() would anchor drives to cwd and collapse UNC authorities.
    return path if is_foreign_windows_path(raw) else path.resolve()


def ambient_workspace() -> Workspace:
    """Fallback workspace for a caller with no session, rooted at the process cwd."""
    return Workspace.for_session(Path.cwd())


def _resolve_display_target(path_str: str, cwd: Path) -> Path | None:
    """Resolve user-provided absolute, relative, or home-relative paths."""
    try:
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = cwd / path
        return Path(os.path.normpath(path))
    except (ValueError, OSError):
        return None


def _display_relative_to_cwd(path: Path, cwd: Path) -> str | None:
    """Return a stable cwd-relative display path, or None when outside cwd."""
    try:
        rel = path.relative_to(cwd)
    except ValueError:
        return None
    if str(rel) == ".":
        return path.name
    return str(rel)


def display_file_path(path_str: str) -> str:
    """Path relative to the session cwd, for display.

    Falls back to the original string when the path can't be resolved, or the
    resolved absolute path when it does not sit under the session cwd.
    """
    manager = _active_file_display_harness.get()
    cwd_raw = (
        (manager.cwd or Path.cwd()).expanduser() if manager is not None else Path.cwd()
    )
    cwd = Path(os.path.normpath(str(cwd_raw)))
    path = _resolve_display_target(path_str, cwd)
    if path is None:
        return path_str

    if relative_path := _display_relative_to_cwd(path, cwd):
        return relative_path
    return str(path)


def _is_windows_path(path: str) -> bool:
    return bool(PureWindowsPath(path).drive) or "\\" in path


def _normalized_path(path: str) -> str:
    if _is_windows_path(path):
        return ntpath.normcase(ntpath.normpath(path))
    return posixpath.normpath(path)


def path_pattern_matches(path: str, pattern: str) -> bool:
    """Match a path glob without letting '*' cross separators when absolute.

    ``fnmatch``'s ``*`` crosses separators, so an allowlist entry like
    ``/home/u/proj/*`` would otherwise authorize the entire subtree below
    ``proj``. Absolute patterns are matched segment-aware (``*`` stops at the
    separator); relative patterns keep requiring the whole path to match
    because ``Path.match`` right-anchors them (``tmp/*`` would otherwise match
    ``/var/tmp/secret``).
    """
    normalized = _normalized_path(path)
    windows = _is_windows_path(path) or _is_windows_path(pattern)
    path_cls = PureWindowsPath if windows else PurePosixPath
    if path_cls(pattern).is_absolute():
        return path_cls(normalized).match(pattern)
    if windows:
        pattern = ntpath.normcase(pattern)
    return fnmatch.fnmatch(normalized, pattern)


def resolve_path_permission(
    path_str: str, *, cwd: Path, allowlist: list[str], denylist: list[str]
) -> PermissionContext | None:
    """Resolve permission for a file path against glob patterns.

    Returns NEVER on denylist match, ALWAYS on allowlist match, None otherwise.
    Allowlist globs are segment-aware: ``*`` in an absolute pattern matches a
    single path level, never a whole subtree.
    """
    if is_foreign_windows_path(path_str):
        return PermissionContext(
            permission=ToolPermission.NEVER,
            reason="Foreign Windows paths are not supported on POSIX",
        )
    file_str = str(_make_absolute(path_str, cwd).resolve())

    for pattern in denylist:
        if fnmatch.fnmatch(file_str, pattern):
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason="File access denied by a configured path rule",
            )

    for pattern in allowlist:
        if path_pattern_matches(file_str, pattern):
            return PermissionContext(permission=ToolPermission.ALWAYS)

    return None


def is_path_within_workdir(
    path_str: str, *, workspace: Workspace | None = None
) -> bool:
    """Return True if the resolved path is inside the workspace's authorised roots.

    Omitting ``workspace`` resolves against the process cwd, not any session's.
    """
    if is_foreign_windows_path(path_str):
        return False
    workspace = workspace or ambient_workspace()
    try:
        resolved = _make_absolute(path_str, workspace.cwd).resolve()
    except (ValueError, OSError):
        return False
    return workspace.allows(resolved)


def resolve_file_tool_permission(  # noqa: PLR0911 - ordered independent denial ceilings
    path_str: str,
    *,
    tool_name: str,
    allowlist: list[str],
    denylist: list[str],
    config_permission: ToolPermission,
    sensitive_patterns: list[str],
    workspace: Workspace | None = None,
    scratchpad_dir: Path | None = None,
    plan_file_write_scope: Path | None = None,
    inherited_plan_write_scopes: tuple[tuple[Path, Path | None], ...] = (),
) -> PermissionContext | None:
    """Resolve permission for a file-based tool invocation.

    Checks unconditional, path and sensitive denies before scoped Plan writes or
    scratchpad. Plan scope is runtime-owned, not an ordinary config permission.
    All other reads and writes require workspace authority; allowlists never
    expand it. Decisions never manufacture per-call approval requirements.
    """
    if config_permission == ToolPermission.NEVER:
        return PermissionContext(
            permission=ToolPermission.NEVER, reason=f"Tool denied: {tool_name}"
        )
    workspace = workspace or ambient_workspace()
    cwd = workspace.cwd
    # Permission checks must classify the same target that file tools execute,
    # not a relative path accidentally anchored to the process working directory.
    file_path = resolve_tool_path(path_str, cwd)
    file_str = str(file_path)

    result = resolve_path_permission(
        file_str, cwd=cwd, allowlist=allowlist, denylist=denylist
    )
    if result is not None and result.permission == ToolPermission.NEVER:
        return result
    if matches_sensitive_pattern(file_str, sensitive_patterns):
        return PermissionContext(
            permission=ToolPermission.NEVER,
            reason=f"Sensitive file access denied ({tool_name})",
        )

    # Every parent scope is a ceiling, never a union of descendant allowances.
    # Session/runtime persistence is not routed through this agent tool grant.
    for parent_plan, parent_scratch in inherited_plan_write_scopes:
        # Ancestor scratch roots are canonicalized when authority is captured.
        # Re-resolution may validate that identity, never redefine its boundary.
        # Targets still resolve normally, allowing safe links into/within scratch.
        # This is a permission check, not a hostile-filesystem race sandbox.
        try:
            in_parent_scratch = (
                parent_scratch is not None
                and parent_scratch.resolve() == parent_scratch
                and file_path.is_relative_to(parent_scratch)
            )
        except (ValueError, OSError):
            in_parent_scratch = False
        if file_path != parent_plan.absolute() and not in_parent_scratch:
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason="Parent Plan scope permits writes only to its designated plan file or session scratchpad",
            )

    in_scratchpad = (
        scratchpad_dir is not None
        and scratchpad_dir.resolve() == scratchpad_dir.absolute()
        and is_scratchpad_path(file_str, scratchpad_dir=scratchpad_dir)
    )
    if plan_file_write_scope is not None or in_scratchpad:
        # Compare the canonical invocation with the designated path, not its
        # symlink target: replacing the plan file with a symlink must not expand
        # write authority to an arbitrary file. No glob expansion is involved.
        allowed = in_scratchpad or (
            plan_file_write_scope is not None
            and file_path == plan_file_write_scope.absolute()
        )
        return PermissionContext(
            permission=ToolPermission.ALWAYS if allowed else ToolPermission.NEVER,
            reason=None
            if allowed
            else "Plan mode permits file writes only to the designated plan file or session scratchpad",
        )

    # The target passed every inherited ceiling and any local Plan scope above.
    # Preserve that exact runtime grant, including a parent's out-of-workspace plan.
    if inherited_plan_write_scopes:
        return PermissionContext(permission=ToolPermission.ALWAYS)

    if not workspace.allows(file_path):
        return PermissionContext(
            permission=ToolPermission.NEVER,
            reason="File access outside authorized project and session scratch roots; only an explicit user scope change can authorize it",
        )

    return PermissionContext(permission=ToolPermission.ALWAYS)
