from __future__ import annotations

from chartreux import CHARTREUX_ROOT
from chartreux.core.session.session_permissions import ensure_private_directory
from chartreux.utils.paths import (
    GlobalPath,
    get_chartreux_home,
    get_chartreux_home_literal,
)

CHARTREUX_HOME = GlobalPath(get_chartreux_home)
GLOBAL_ENV_FILE = GlobalPath(lambda: CHARTREUX_HOME.path / ".env")
SESSION_LOG_DIR = GlobalPath(lambda: CHARTREUX_HOME.path / "logs" / "session")
WORKTREES_DIR = GlobalPath(lambda: CHARTREUX_HOME.path / "worktrees")
TRUSTED_FOLDERS_FILE = GlobalPath(lambda: CHARTREUX_HOME.path / "trusted_folders.toml")
LOG_DIR = GlobalPath(lambda: CHARTREUX_HOME.path / "logs")
LOG_FILE = GlobalPath(lambda: CHARTREUX_HOME.path / "logs" / "chartreux.log")
LITERAL_LOG_DIR = GlobalPath(lambda: get_chartreux_home_literal() / "logs")
LITERAL_LOG_FILE = GlobalPath(
    lambda: get_chartreux_home_literal() / "logs" / "chartreux.log"
)
CACHE_FILE = GlobalPath(lambda: CHARTREUX_HOME.path / "cache.toml")
PROJECTS_FILE = GlobalPath(lambda: CHARTREUX_HOME.path / "projects.toml")
HISTORY_FILE = GlobalPath(lambda: CHARTREUX_HOME.path / "chartreuxhistory")
PLANS_DIR = GlobalPath(lambda: CHARTREUX_HOME.path / "plans")


def ensure_chartreux_home_private() -> None:
    """Create or repair the application-owned home without changing its parent."""
    ensure_private_directory(get_chartreux_home_literal())


DEFAULT_TOOL_DIR = GlobalPath(lambda: CHARTREUX_ROOT / "core" / "tools" / "builtins")
