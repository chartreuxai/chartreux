"""Project-env injection: the trusted folder's ``.env`` and ``$CHARTREUX_HOME/.env``.

When a bash call's working folder is inside a trusted folder, exactly two
dotenv files load into the bash tool's subprocess environment:

1. the trusted folder's ``.env`` (project level);
2. ``$CHARTREUX_HOME/.env`` (global level).

Project-level keys win over global-level keys. There are no profile variants
and no cascading lookups: the trusted folder is the trust root of the working
folder, and only its own ``.env`` loads — a working folder nested deeper does
not pick up intermediate ``.env`` files. Both files are re-read when their
mtime changes, so secret rotation takes effect on the next command without a
restart; the session redaction set follows the same reload.
"""

from __future__ import annotations

from pathlib import Path
import threading

from dotenv import dotenv_values

from chartreux.core.paths import GLOBAL_ENV_FILE
from chartreux.core.trusted_folders import (
    TrustedFoldersManager,
    trusted_folders_manager,
)

_lock = threading.Lock()
#: Parsed entries per dotenv file, keyed by resolved path and re-read when the
#: file's identity token (device, inode, ctime, mtime, size) changes.
_entries_cache: dict[Path, tuple[object, dict[str, str]]] = {}


def _file_token(path: Path) -> object:
    """A cheap identity token for a dotenv file, ``None`` when absent."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_ctime_ns, stat.st_mtime_ns, stat.st_size)


def dotenv_file_entries(path: Path) -> dict[str, str]:
    """Parse one dotenv file into its non-empty ``name -> value`` entries.

    The real dotenv parser handles quoting, ``export`` prefixes, comments, and
    multiline values. Results are cached per file and re-read when the file
    changes on disk. Fail-closed: an absent or unreadable file yields no
    entries, and assignments without a value never inject. Only regular files
    load: a FIFO or other non-regular file is treated as absent, so a FIFO
    ``.env`` can never block env construction on a blocking open.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return {}
    if not resolved.is_file():
        return {}
    token = _file_token(resolved)
    if token is None:
        return {}
    with _lock:
        cached = _entries_cache.get(resolved)
        if cached is not None and cached[0] == token:
            return dict(cached[1])
    try:
        raw = dotenv_values(resolved)
    except Exception:
        return {}
    entries = {
        name: value for name, value in raw.items() if isinstance(value, str) and value
    }
    with _lock:
        _entries_cache[resolved] = (token, entries)
    return dict(entries)


def trusted_folder_for(
    cwd: Path, *, manager: TrustedFoldersManager | None = None
) -> Path | None:
    """The trusted folder a working folder belongs to, ``None`` when untrusted."""
    manager = manager or trusted_folders_manager
    return manager.find_trust_root(cwd)


def project_env_file(
    cwd: Path, *, manager: TrustedFoldersManager | None = None
) -> Path | None:
    """The trusted folder's ``.env`` for *cwd*, ``None`` outside trusted folders."""
    root = trusted_folder_for(cwd, manager=manager)
    return None if root is None else root / ".env"


def load_project_env(trusted_folder: Path) -> dict[str, str]:
    """Load exactly two dotenv files; project-level keys win.

    ``$CHARTREUX_HOME/.env`` loads first and the trusted folder's ``.env``
    second, so the project's rotation wins per key.
    """
    entries = dotenv_file_entries(GLOBAL_ENV_FILE.path)
    entries.update(dotenv_file_entries(trusted_folder / ".env"))
    return entries


def project_env_for_cwd(
    cwd: Path | None, *, manager: TrustedFoldersManager | None = None
) -> dict[str, str]:
    """The merged project environment for a bash call's working folder.

    Empty when the working folder is not inside a trusted folder.
    """
    root = trusted_folder_for(cwd or Path.cwd(), manager=manager)
    if root is None:
        return {}
    return load_project_env(root)


def reset_cache() -> None:
    """Drop cached dotenv entries (tests)."""
    with _lock:
        _entries_cache.clear()


__all__ = [
    "dotenv_file_entries",
    "load_project_env",
    "project_env_file",
    "project_env_for_cwd",
    "reset_cache",
    "trusted_folder_for",
]
