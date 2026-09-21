from __future__ import annotations

import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from chartreux.acp.acp_logger import ACP_LOG_DIR, ACP_LOG_FILE
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.config.layers.environment import EnvironmentLayer
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.paths import CHARTREUX_HOME, HISTORY_FILE, LOG_FILE, SESSION_LOG_DIR
from chartreux.core.paths._local_config_files import find_local_config_dirs
from chartreux.utils.paths import get_chartreux_home


@pytest.mark.asyncio
async def test_legacy_environment_prefix_is_ignored_with_or_without_new_value() -> None:
    with patch.dict(os.environ, {"VIBE_ACTIVE_MODEL": "legacy"}, clear=True):
        without_new = await EnvironmentLayer(schema=ChartreuxConfigSchema).load()
    with patch.dict(
        os.environ,
        {"VIBE_ACTIVE_MODEL": "legacy", "CHARTREUX_ACTIVE_MODEL": "current"},
        clear=True,
    ):
        with_new = await EnvironmentLayer(schema=ChartreuxConfigSchema).load()

    assert without_new.model_dump() == {}
    assert with_new.model_dump() == {"active_model": "current"}


def test_legacy_home_environment_is_ignored_and_old_tree_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default_home = tmp_path / "default" / ".chartreux"
    new_home = tmp_path / "new" / ".chartreux"
    old_home = tmp_path / "old" / ".vibe"
    old_home.mkdir(parents=True)
    sentinel = old_home / "sentinel.txt"
    sentinel.write_text("legacy", encoding="utf-8")
    monkeypatch.setattr("chartreux.utils.paths._DEFAULT_CHARTREUX_HOME", default_home)
    monkeypatch.setenv("VIBE_HOME", str(old_home))
    monkeypatch.delenv("CHARTREUX_HOME", raising=False)

    assert get_chartreux_home() == default_home
    monkeypatch.setenv("CHARTREUX_HOME", str(new_home))
    assert get_chartreux_home() == new_home.resolve()
    assert sentinel.read_text(encoding="utf-8") == "legacy"
    assert not new_home.exists()


def test_project_discovery_uses_chartreux_and_retains_generic_agents(
    tmp_path: Path,
) -> None:
    old_dir = tmp_path / ".vibe" / "tools"
    chartreux_dir = tmp_path / ".chartreux" / "tools"
    agents_dir = tmp_path / ".agents" / "skills"
    old_dir.mkdir(parents=True)
    chartreux_dir.mkdir(parents=True)
    agents_dir.mkdir(parents=True)

    result = find_local_config_dirs(tmp_path)

    assert old_dir not in result.tools
    assert tmp_path / ".vibe" not in result.config_dirs
    assert chartreux_dir in result.tools
    assert tmp_path / ".chartreux" in result.config_dirs
    assert agents_dir in result.skills
    assert tmp_path / ".agents" in result.config_dirs


def test_runtime_paths_and_acp_logging_follow_chartreux_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".chartreux"
    monkeypatch.setenv("CHARTREUX_HOME", str(home))
    monkeypatch.setenv("VIBE_HOME", str(tmp_path / "legacy-home"))

    assert CHARTREUX_HOME.path == home
    assert LOG_FILE.path == home / "logs" / "chartreux.log"
    assert SESSION_LOG_DIR.path == home / "logs" / "session"
    assert HISTORY_FILE.path == home / "chartreuxhistory"
    assert ACP_LOG_DIR.path == home / "logs" / "acp"
    assert ACP_LOG_FILE.path == home / "logs" / "acp" / "messages.jsonl"

    manager = HarnessFilesManager(sources=("user",), cwd=home.parent)
    assert manager.config_file == home / "config.toml"
    assert manager.user_config_file == home / "config.toml"
    assert manager.cwd_is_user_config_home is True


@pytest.mark.asyncio
async def test_project_config_does_not_fall_back_to_legacy_vibe_config(
    tmp_path: Path,
) -> None:
    legacy_config = tmp_path / ".vibe" / "config.toml"
    legacy_config.parent.mkdir(parents=True)
    legacy_config.write_text('active_model = "legacy-model"\n', encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()

    layer = ProjectConfigLayer(path=project)
    data = await layer.load()

    assert data.model_extra == {}
    assert layer.config_file_path is None


@pytest.mark.parametrize(
    ("relative_path", "ignored"),
    [
        (".vibe/config.synthetic.toml", True),
        (".vibe/skills/untracked-synthetic/SKILL.md", True),
        (".vibe/skills/create-vibe-feature/synthetic.txt", True),
        (".chartreux/.env", True),
        (".chartreux/logs/synthetic.log", True),
        (".chartreux/skills/untracked-synthetic/SKILL.md", True),
        (".chartreux/skills/create-vibe-feature/synthetic.txt", False),
    ],
)
def test_runtime_ignore_rules_cover_both_identities_without_creating_files(
    relative_path: str, ignored: bool
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    synthetic_path = repo_root / relative_path
    assert not synthetic_path.exists()

    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", "--", relative_path],
        cwd=repo_root,
        check=False,
    )

    assert (result.returncode == 0) is ignored
