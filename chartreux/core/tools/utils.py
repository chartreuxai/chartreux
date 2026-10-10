from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
import fnmatch
import ntpath
import os
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
import posixpath

from chartreux.core.config.harness_files import HarnessFilesManager
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


_instruction_read_ceiling: ContextVar[frozenset[Path] | None] = ContextVar(
    "instruction_read_ceiling", default=None
)


@contextmanager
def instruction_read_ceiling(files: frozenset[Path]) -> Iterator[None]:
    """Apply the requesting agent's manifest while checking ancestor denials.

    Ancestors still enforce every NEVER ceiling; their workspace boundary alone
    must not suppress a document injected into the requesting child's context.
    Nested ancestors retain the original request manifest.
    """
    existing = _instruction_read_ceiling.get()
    token = _instruction_read_ceiling.set(existing if existing is not None else files)
    try:
        yield
    finally:
        _instruction_read_ceiling.reset(token)


_scratchpad_ceiling: ContextVar[frozenset[Path] | None] = ContextVar(
    "scratchpad_ceiling", default=None
)


@contextmanager
def scratchpad_ceiling(roots: frozenset[Path]) -> Iterator[None]:
    """Carry the original request's live grants through ancestor denial checks."""
    existing = _scratchpad_ceiling.get()
    token = _scratchpad_ceiling.set(existing if existing is not None else roots)
    try:
        yield
    finally:
        _scratchpad_ceiling.reset(token)


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
            denial_kind="path_rule",
        )
    file_str = str(_make_absolute(path_str, cwd).resolve())

    for pattern in denylist:
        if fnmatch.fnmatch(file_str, pattern):
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason="File access denied by a configured path rule",
                denial_kind="path_rule",
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


def is_canonical_scratch_path(file_path: Path, root: Path | None) -> bool:
    """Check a pinned root without redefining its boundary after symlink mutation."""
    try:
        return (
            root is not None
            and root.resolve() == root.absolute()
            and file_path.is_relative_to(root.absolute())
        )
    except (ValueError, OSError):
        return False


class PathAccess(StrEnum):
    READ = "read"
    WRITE = "write"
    MUTATE = "mutate"


@dataclass(frozen=True)
class PathAuthority:
    """Immutable path-policy inputs shared by file tools and shell operands.

    A snapshot is an optimization, not an approval: callers still validate the
    manager token and preserve custom invocation resolvers.
    """

    tool_name: str
    permission: ToolPermission
    allowlist: tuple[str, ...]
    denylist: tuple[str, ...]
    sensitive: tuple[str, ...]
    workspace: Workspace
    scratchpad: Path | None = None
    scratchpad_roots: frozenset[Path] = frozenset()
    instruction_read_files: frozenset[Path] = frozenset()
    parents: tuple[PathAuthority, ...] = ()
    plan_file_write_scope: Path | None = None
    inherited_plan_write_scopes: tuple[tuple[Path, Path | None], ...] = ()

    def resolve(self, path: str, access: PathAccess) -> PermissionContext | None:
        decision = resolve_file_tool_permission(
            path,
            tool_name=self.tool_name,
            allowlist=list(self.allowlist),
            denylist=list(self.denylist),
            config_permission=self.permission,
            sensitive_patterns=list(self.sensitive),
            workspace=self.workspace,
            scratchpad_dir=self.scratchpad,
            scratchpad_roots=self.scratchpad_roots,
            instruction_read_files=self.instruction_read_files,
            plan_file_write_scope=self.plan_file_write_scope,
            inherited_plan_write_scopes=self.inherited_plan_write_scopes,
            access=access,
        )
        for parent in self.parents:
            inherited = parent.resolve(path, access)
            if inherited is not None and inherited.permission == ToolPermission.NEVER:
                if inherited.denial_kind == "out_of_root":
                    return inherited.model_copy(
                        update={"denial_kind": "parent_ceiling"}
                    )
                return inherited
        return decision

    def allows(self, path: Path, access: PathAccess = PathAccess.READ) -> bool:
        decision = self.resolve(str(path), access)
        return decision is not None and decision.permission == ToolPermission.ALWAYS


def resolve_file_tool_permission(  # noqa: PLR0911, PLR0912, PLR0913 - independent runtime denial ceilings
    path_str: str,
    *,
    tool_name: str,
    allowlist: list[str],
    denylist: list[str],
    config_permission: ToolPermission,
    sensitive_patterns: list[str],
    workspace: Workspace | None = None,
    scratchpad_dir: Path | None = None,
    scratchpad_roots: frozenset[Path] = frozenset(),
    plan_file_write_scope: Path | None = None,
    inherited_plan_write_scopes: tuple[tuple[Path, Path | None], ...] = (),
    instruction_read_files: frozenset[Path] = frozenset(),
    access: PathAccess | None = None,
) -> PermissionContext | None:
    """Resolve permission for a file-based tool invocation.

    Checks unconditional, path and sensitive denies before scoped Plan writes or
    scratchpad. Plan scope is runtime-owned, not an ordinary config permission.
    Exact injected instruction files additionally permit read_file/grep and
    explicitly modeled shell reads, never writes or opaque mutators.
    Other reads and writes require workspace authority; allowlists never expand
    it. Decisions never manufacture per-call approval requirements.
    """
    if config_permission == ToolPermission.NEVER:
        return PermissionContext(
            permission=ToolPermission.NEVER,
            reason=f"Tool denied: {tool_name}",
            denial_kind="tool_policy",
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
            denial_kind="sensitive",
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
                denial_kind="plan_scope",
            )

    roots = scratchpad_roots
    if scratchpad_dir is not None:
        roots |= {scratchpad_dir.absolute()}
    if (request_roots := _scratchpad_ceiling.get()) is not None:
        roots = request_roots
    in_scratchpad = any(is_canonical_scratch_path(file_path, root) for root in roots)
    if plan_file_write_scope is not None:
        # A request grant cannot expand this ancestor's own Plan ceiling.
        allowed = is_canonical_scratch_path(file_path, scratchpad_dir) or (
            file_path == plan_file_write_scope.absolute()
        )
        return PermissionContext(
            permission=ToolPermission.ALWAYS if allowed else ToolPermission.NEVER,
            reason=None
            if allowed
            else "Plan mode permits file writes only to the designated plan file or session scratchpad",
            denial_kind=None if allowed else "plan_scope",
        )

    if (
        tool_name in {"bash", "bash_start"}
        and is_canonical_scratch_path(file_path, scratchpad_dir)
        and workspace.ceiling is not None
        and not workspace.ceiling.allows(file_path)
    ):
        return PermissionContext(
            permission=ToolPermission.NEVER,
            reason="File access outside authorized project and session scratch roots; only an explicit user scope change can authorize it",
            denial_kind="plan_scope",
        )
    if in_scratchpad:
        return PermissionContext(permission=ToolPermission.ALWAYS)

    # The target passed every inherited ceiling and any local Plan scope above.
    # Preserve that exact runtime grant, including a parent's out-of-workspace plan.
    if inherited_plan_write_scopes:
        return PermissionContext(permission=ToolPermission.ALWAYS)

    # Exact-file read capability, never an effect-kind or directory grant.
    # Canonical identities were pinned by the loader; do not re-resolve them.
    read_only = (
        access == PathAccess.READ
        and tool_name in {"read_file", "grep", "bash", "bash_start"}
        if access is not None
        else tool_name in {"read_file", "grep"}
    )
    files = instruction_read_files
    if read_only and (request_files := _instruction_read_ceiling.get()) is not None:
        files = request_files
    if read_only and file_path in files and file_path.is_file():
        # Request-time check, not a race sandbox: post-injection mutation is accepted, as with prompt refresh.
        return PermissionContext(permission=ToolPermission.ALWAYS)

    if not workspace.allows(file_path):
        reason = (
            "File read is outside authorized workspace and session scratch roots "
            "and is not an instruction file injected into this agent's context. "
            "Other paths require an explicit user scope change."
            if read_only
            else "Injected instruction files are readable only; writes require an explicit user scope change."
            if tool_name in {"write_file", "edit"} and file_path in files
            else "File access outside authorized project and session scratch roots; only an explicit user scope change can authorize it"
        )
        own_roots_allow = False
        try:
            own_roots_allow = any(
                root.resolve() == root and file_path.is_relative_to(root)
                for root in workspace.authorized_roots
            )
        except (ValueError, OSError):
            own_roots_allow = False
        ceiling_allows = workspace.ceiling is None or workspace.ceiling.allows(
            file_path
        )
        return PermissionContext(
            permission=ToolPermission.NEVER,
            reason=reason,
            denial_kind="out_of_root"
            if not own_roots_allow and ceiling_allows
            else "parent_ceiling",
        )

    return PermissionContext(permission=ToolPermission.ALWAYS)
