from __future__ import annotations

import os
from pathlib import Path
import platform
import sys
from typing import Final

_GIT_EXECUTABLE_ENV: Final = "GIT_PYTHON_GIT_EXECUTABLE"
_PROCESS_GIT_OVERRIDE: Final[str | None] = os.environ.get(_GIT_EXECUTABLE_ENV)

_PLATFORM_IDS: Final[dict[str, str]] = {
    "darwin": "darwin",
    "linux": "linux",
    "freebsd": "freebsd",
    "openbsd": "openbsd",
    "netbsd": "netbsd",
}

_PLATFORM_DISPLAY_NAMES: Final[dict[str, str]] = {
    "darwin": "macOS",
    "linux": "Linux",
    "freebsd": "FreeBSD",
    "openbsd": "OpenBSD",
    "netbsd": "NetBSD",
}


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and (os.name == "nt" or os.access(path, os.X_OK))


def _resolved_executable(path: Path) -> Path | None:
    try:
        resolved = path.expanduser().resolve()
        return resolved if _is_executable_file(resolved) else None
    except (OSError, RuntimeError):
        return None


def _looks_like_bare_repository(path: Path) -> bool:
    try:
        return (
            (path / "HEAD").is_file()
            and (path / "objects").is_dir()
            and (path / "refs").is_dir()
        )
    except OSError:
        raise


def _resolved_directory_env(value: str, *, current: Path) -> Path | None:
    try:
        configured = Path(value).expanduser()
        if not configured.is_absolute():
            configured = current / configured
        resolved = configured.resolve(strict=True)
        return resolved if resolved.is_dir() else None
    except (OSError, RuntimeError):
        return None


def _repository_boundaries(path: Path) -> tuple[Path, ...] | None:
    """Return every repository boundary containing *path*, or fail closed."""
    current = path if path.is_dir() else path.parent
    boundaries: list[Path] = []
    for candidate in (current, *current.parents):
        marker = candidate / ".git"
        try:
            if marker.is_file() or marker.is_dir():
                boundaries.append(candidate)
            elif marker.exists() or marker.is_symlink():
                return None
            if _looks_like_bare_repository(candidate):
                boundaries.append(candidate)
        except OSError:
            return None

    configured_git_dir = os.environ.get("GIT_DIR")
    if configured_git_dir:
        git_dir = _resolved_directory_env(configured_git_dir, current=current)
        if git_dir is None:
            return None
        boundaries.append(git_dir.parent if git_dir.name == ".git" else git_dir)

    configured_work_tree = os.environ.get("GIT_WORK_TREE")
    if configured_work_tree:
        work_tree = _resolved_directory_env(configured_work_tree, current=current)
        if work_tree is None:
            return None
        boundaries.append(work_tree)
    return tuple(dict.fromkeys(boundaries))


def _is_untrusted_project_executable(path: Path, cwd: Path) -> bool:
    boundaries = _repository_boundaries(cwd)
    if boundaries is None:
        return True
    project_dirs = tuple(dict.fromkeys((*boundaries, cwd)))
    for project_dir in project_dirs:
        try:
            path.relative_to(project_dir)
        except ValueError:
            continue

        # Filesystem roots and home directories contain normal executable locations.
        # In those broad directories, only an executable directly in the directory is
        # the implicit current-directory candidate that automatic discovery must skip.
        if project_dir.parent == project_dir:
            if path.parent == project_dir:
                return True
            continue
        try:
            if project_dir == Path.home().expanduser().resolve():
                if path.parent == project_dir:
                    return True
                continue
        except (OSError, RuntimeError):
            return True
        return True
    return False


def _executable_names(name: str) -> tuple[str, ...]:
    if os.name != "nt" or Path(name).suffix:
        return (name,)
    extensions = os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(os.pathsep)
    return tuple(f"{name}{extension.lower()}" for extension in extensions if extension)


def _search_trusted_path(name: str, *, cwd: Path) -> str | None:
    """Resolve an executable without relative or implicit-current-dir search."""
    for raw_entry in os.get_exec_path():
        entry = Path(raw_entry.strip('"')).expanduser()
        if not entry.is_absolute():
            continue
        for executable_name in _executable_names(name):
            candidate = _resolved_executable(entry / executable_name)
            if candidate is not None and not _is_untrusted_project_executable(
                candidate, cwd
            ):
                return str(candidate)
    return None


def _resolve_configured_git(configured: str) -> str | None:
    configured_path = Path(configured).expanduser()
    if not configured_path.is_absolute():
        return None
    resolved = _resolved_executable(configured_path)
    return str(resolved) if resolved is not None else None


def resolve_git_executable(*, cwd: Path | None = None) -> str | None:
    """Return an absolute Git executable safe for application-owned calls.

    An absolute process-level ``GIT_PYTHON_GIT_EXECUTABLE`` is an explicit trust
    decision. Automatic discovery ignores relative PATH entries and executables
    located in the project, including candidates whose symlink target is there.
    """
    try:
        project_dir = (cwd or Path.cwd()).resolve()
    except (OSError, RuntimeError):
        return None

    configured = os.environ.get(_GIT_EXECUTABLE_ENV)
    if configured and configured == _PROCESS_GIT_OVERRIDE:
        return _resolve_configured_git(configured)
    if configured:
        pinned = _resolved_executable(Path(configured))
        if pinned is not None and not _is_untrusted_project_executable(
            pinned, project_dir
        ):
            return str(pinned)
    return _search_trusted_path("git", cwd=project_dir)


def resolve_ssh_executable(*, cwd: Path | None = None) -> str | None:
    """Return an absolute SSH client that is not a project-local binary.

    Automatic discovery never searches relative PATH entries or selects an
    executable from the working directory.
    """
    try:
        project_dir = (cwd or Path.cwd()).resolve()
    except (OSError, RuntimeError):
        return None
    return _search_trusted_path("ssh", cwd=project_dir)


def configure_git_python_executable(*, cwd: Path | None = None) -> str | None:
    """Pin GitPython to the same trusted executable used by direct callers."""
    executable = resolve_git_executable(cwd=cwd)
    if executable is not None:
        configured = os.environ.get(_GIT_EXECUTABLE_ENV)
        if _PROCESS_GIT_OVERRIDE is None or configured != _PROCESS_GIT_OVERRIDE:
            os.environ[_GIT_EXECUTABLE_ENV] = executable
    return executable


def get_platform_id() -> str:
    """Canonical lowercase platform identifier (e.g. ``darwin``, ``linux``).

    Suitable for machine-readable contexts. Falls back to the raw ``sys.platform``
    value for unknown platforms.
    """
    return _PLATFORM_IDS.get(sys.platform, sys.platform)


def get_platform_version() -> str | None:
    match get_platform_id():
        case "darwin":
            version = platform.mac_ver()[0] or platform.release()
        case "linux":
            version = _linux_os_version() or platform.release()
        case _:
            version = platform.release() or platform.version()
    return version or None


def _linux_os_version() -> str | None:
    try:
        os_release = platform.freedesktop_os_release()
    except OSError:
        return None
    return os_release.get("VERSION_ID") or os_release.get("VERSION")


def get_platform_display_name() -> str:
    """Human-readable platform name (e.g. ``macOS``, ``Linux``).

    Suitable for surfacing in system prompts. Falls back to ``Unix-like`` for
    unknown platforms.
    """
    return _PLATFORM_DISPLAY_NAMES.get(get_platform_id(), "Unix-like")
