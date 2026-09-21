from __future__ import annotations

from chartreux.core.paths import AGENTS_HOME, CHARTREUX_HOME, GlobalPath

GLOBAL_TOOLS_DIR = GlobalPath(lambda: CHARTREUX_HOME.path / "tools")
GLOBAL_SKILLS_DIR = GlobalPath(lambda: CHARTREUX_HOME.path / "skills")
GLOBAL_AGENTS_DIR = GlobalPath(lambda: CHARTREUX_HOME.path / "agents")
GLOBAL_PROMPTS_DIR = GlobalPath(lambda: CHARTREUX_HOME.path / "prompts")
GLOBAL_AGENTS_SKILLS_DIR = GlobalPath(lambda: AGENTS_HOME.path / "skills")
