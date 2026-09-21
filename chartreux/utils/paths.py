from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path, PureWindowsPath
from urllib.parse import unquote, urlparse


def file_uri_to_path(uri: str) -> str:
    """Convert a file URI into a path using the current platform's rules."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Expected a file URI: {uri!r}")

    encoded_path = parsed.path
    if parsed.netloc and parsed.netloc != "localhost":
        encoded_path = f"//{parsed.netloc}{encoded_path}"
    return unquote(encoded_path)


def is_foreign_windows_path(raw: str) -> bool:
    """Return whether ``raw`` is an anchored Windows path on this POSIX host."""
    path = PureWindowsPath(raw.strip())
    return bool(path.drive and path.root)


class GlobalPath:
    def __init__(self, resolver: Callable[[], Path]) -> None:
        self._resolver = resolver

    @property
    def path(self) -> Path:
        return self._resolver()


_DEFAULT_CHARTREUX_HOME = Path.home() / ".chartreux"


def get_chartreux_home_literal() -> Path:
    """Return the configured home spelling without resolving symlinks."""
    if chartreux_home := os.getenv("CHARTREUX_HOME"):
        return Path(chartreux_home).expanduser()
    return _DEFAULT_CHARTREUX_HOME


def get_chartreux_home() -> Path:
    """Return the canonical application home for identity and boundary checks."""
    return get_chartreux_home_literal().resolve()


def is_dangerous_directory(path: Path | str = ".") -> tuple[bool, str]:
    """Check if the current directory is a dangerous folder that would cause
    issues if we were to run the tool there.

    Args:
        path: Path to check (defaults to current directory)

    Returns:
        tuple[bool, str]: (is_dangerous, reason) where reason explains why it's dangerous
    """
    path = Path(path).resolve()

    home_dir = Path.home()

    dangerous_paths = {
        home_dir: "home directory",
        home_dir / "Documents": "Documents folder",
        home_dir / "Desktop": "Desktop folder",
        home_dir / "Downloads": "Downloads folder",
        home_dir / "Pictures": "Pictures folder",
        home_dir / "Movies": "Movies folder",
        home_dir / "Music": "Music folder",
        home_dir / "Library": "Library folder",
        Path("/Applications"): "Applications folder",
        Path("/System"): "System folder",
        Path("/Library"): "System Library folder",
        Path("/usr"): "System usr folder",
        Path("/private"): "System private folder",
    }

    for dangerous_path, description in dangerous_paths.items():
        try:
            if path == dangerous_path:
                return True, f"You are in the {description}"
        except (OSError, ValueError):
            continue
    return False, ""
