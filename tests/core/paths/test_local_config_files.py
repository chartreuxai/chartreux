from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.paths._local_config_files import (
    LocalConfigDirs,
    find_local_config_dirs,
)


class TestSubdirs:
    def test_finds_config_at_root(self, tmp_path: Path) -> None:
        (tmp_path / ".chartreux" / "tools").mkdir(parents=True)
        result = find_local_config_dirs(tmp_path)
        assert tmp_path.resolve() / ".chartreux" / "tools" in result.tools

    def test_does_not_descend_into_subdirectories(self, tmp_path: Path) -> None:
        (tmp_path / "sub" / ".chartreux" / "tools").mkdir(parents=True)
        (tmp_path / "a" / "b" / ".chartreux" / "skills").mkdir(parents=True)
        result = find_local_config_dirs(tmp_path)
        assert result.tools == ()
        assert result.skills == ()
        assert result.agents == ()
        assert result.config_dirs == ()

    def test_finds_agents_skills_at_root(self, tmp_path: Path) -> None:
        (tmp_path / ".agents" / "skills").mkdir(parents=True)
        result = find_local_config_dirs(tmp_path)
        assert tmp_path.resolve() / ".agents" / "skills" in result.skills

    def test_finds_all_config_types_at_root(self, tmp_path: Path) -> None:
        (tmp_path / ".chartreux" / "tools").mkdir(parents=True)
        (tmp_path / ".chartreux" / "skills").mkdir(parents=True)
        (tmp_path / ".chartreux" / "agents").mkdir(parents=True)
        (tmp_path / ".agents" / "skills").mkdir(parents=True)
        result = find_local_config_dirs(tmp_path)
        resolved = tmp_path.resolve()
        assert resolved / ".chartreux" / "tools" in result.tools
        assert resolved / ".chartreux" / "skills" in result.skills
        assert resolved / ".chartreux" / "agents" in result.agents
        assert resolved / ".agents" / "skills" in result.skills


class TestConfigDirs:
    def test_ignores_plugin_only_directory_without_touching_it(
        self, tmp_path: Path
    ) -> None:
        directory = tmp_path / ".chartreux" / "plugins" / "example"
        directory.mkdir(parents=True)
        sentinel = directory / "plugin.json"
        sentinel.write_text('{"name": "example"}')

        result = find_local_config_dirs(tmp_path)

        assert result == LocalConfigDirs()
        assert not hasattr(result, "plugins")
        assert sentinel.read_text() == '{"name": "example"}'

    def test_plugins_do_not_hide_retained_local_extensions(
        self, tmp_path: Path
    ) -> None:
        for name in ("plugins", "tools", "skills", "agents"):
            (tmp_path / ".chartreux" / name).mkdir(parents=True)
        (tmp_path / ".agents" / "skills").mkdir(parents=True)

        result = find_local_config_dirs(tmp_path)
        root = tmp_path.resolve()

        assert result == LocalConfigDirs(
            config_dirs=(root / ".chartreux", root / ".agents"),
            tools=(root / ".chartreux" / "tools",),
            skills=(root / ".chartreux" / "skills", root / ".agents" / "skills"),
            agents=(root / ".chartreux" / "agents",),
        )
        assert (root / ".chartreux" / "plugins").is_dir()

    def test_finds_vibe_with_tools(self, tmp_path: Path) -> None:
        (tmp_path / ".chartreux" / "tools").mkdir(parents=True)
        result = find_local_config_dirs(tmp_path)
        assert tmp_path.resolve() / ".chartreux" in result.config_dirs

    def test_finds_vibe_with_skills(self, tmp_path: Path) -> None:
        (tmp_path / ".chartreux" / "skills").mkdir(parents=True)
        result = find_local_config_dirs(tmp_path)
        assert tmp_path.resolve() / ".chartreux" in result.config_dirs

    def test_finds_agents_with_skills(self, tmp_path: Path) -> None:
        (tmp_path / ".agents" / "skills").mkdir(parents=True)
        result = find_local_config_dirs(tmp_path)
        assert tmp_path.resolve() / ".agents" in result.config_dirs

    def test_ignores_empty_vibe_dir(self, tmp_path: Path) -> None:
        (tmp_path / ".chartreux").mkdir()
        result = find_local_config_dirs(tmp_path)
        assert result.config_dirs == ()

    def test_ignores_empty_agents_dir(self, tmp_path: Path) -> None:
        (tmp_path / ".agents").mkdir()
        result = find_local_config_dirs(tmp_path)
        assert result.config_dirs == ()

    def test_returns_empty_when_empty(self, tmp_path: Path) -> None:
        result = find_local_config_dirs(tmp_path)
        assert result.config_dirs == ()

    def test_finds_vibe_with_prompts(self, tmp_path: Path) -> None:
        (tmp_path / ".chartreux" / "prompts").mkdir(parents=True)
        result = find_local_config_dirs(tmp_path)
        assert tmp_path.resolve() / ".chartreux" in result.config_dirs

    def test_finds_vibe_with_config_toml(self, tmp_path: Path) -> None:
        (tmp_path / ".chartreux").mkdir()
        (tmp_path / ".chartreux" / "config.toml").write_text("")
        result = find_local_config_dirs(tmp_path)
        assert tmp_path.resolve() / ".chartreux" in result.config_dirs

    def test_finds_vibe_and_agents_at_same_root(self, tmp_path: Path) -> None:
        (tmp_path / ".chartreux" / "skills").mkdir(parents=True)
        (tmp_path / ".agents" / "skills").mkdir(parents=True)
        result = find_local_config_dirs(tmp_path)
        resolved = tmp_path.resolve()
        assert resolved / ".chartreux" in result.config_dirs
        assert resolved / ".agents" in result.config_dirs

    def test_unreadable_config_dirs_do_not_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_is_dir(self: Path) -> bool:
            raise PermissionError(13, "Permission denied")

        def fake_is_file(self: Path) -> bool:
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(Path, "is_dir", fake_is_dir)
        monkeypatch.setattr(Path, "is_file", fake_is_file)

        result = find_local_config_dirs(tmp_path)
        assert result == LocalConfigDirs()


class TestLocalConfigDirsOr:
    def test_or_concatenates_each_field(self) -> None:
        a = LocalConfigDirs(
            config_dirs=(Path("/a/.chartreux"),),
            tools=(Path("/a/.chartreux/tools"),),
            skills=(Path("/a/.chartreux/skills"),),
            agents=(Path("/a/.chartreux/agents"),),
        )
        b = LocalConfigDirs(
            config_dirs=(Path("/b/.chartreux"),),
            tools=(Path("/b/.chartreux/tools"),),
            skills=(Path("/b/.chartreux/skills"),),
            agents=(Path("/b/.chartreux/agents"),),
        )
        merged = a | b
        assert merged.config_dirs == (Path("/a/.chartreux"), Path("/b/.chartreux"))
        assert merged.tools == (
            Path("/a/.chartreux/tools"),
            Path("/b/.chartreux/tools"),
        )
        assert merged.skills == (
            Path("/a/.chartreux/skills"),
            Path("/b/.chartreux/skills"),
        )
        assert merged.agents == (
            Path("/a/.chartreux/agents"),
            Path("/b/.chartreux/agents"),
        )

    def test_or_with_empty_is_identity(self) -> None:
        a = LocalConfigDirs(tools=(Path("/a/.chartreux/tools"),))
        assert (a | LocalConfigDirs()) == a
        assert (LocalConfigDirs() | a) == a
