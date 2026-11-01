"""K1 acceptance: shadow inventory, doctor warn status, `chartreux skills diff`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chartreux.cli import doctor_command as doctor, skills_command
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.harness_files import (
    HarnessFilesManager,
    get_harness_files_manager,
)
from chartreux.core.skills import manager as skills_manager
from chartreux.core.skills.manager import (
    ShadowEntry,
    ShadowInventory,
    ShadowState,
    build_shadow_inventory,
)
from chartreux.core.trusted_folders import TrustedFoldersManager
from tests.conftest import build_test_vibe_config

SHIPPED_SKILLS = skills_manager.SHIPPED_SKILLS_DIR
SKILL = "sub-finder"
ENTRY_KEYS = {
    "name",
    "path",
    "user_dir",
    "state",
    "shipped_path",
    "winner_path",
    "winner_scope",
    "diff",
    "reason",
}

IDENTICAL_WORDING = "suggest removal after reviewing support files — never automatic"


@pytest.fixture
def user_skills(config_dir: Path) -> Path:
    """The first USER skill directory (~/.chartreux/skills)."""
    path = config_dir / "skills"
    path.mkdir()
    return path


@pytest.fixture
def agents_skills(config_dir: Path) -> Path:
    """The second USER skill directory (~/.agents/skills)."""
    path = config_dir.parent / ".agents" / "skills"
    path.mkdir(parents=True)
    return path


def shipped_text(name: str = SKILL) -> str:
    return (SHIPPED_SKILLS / name / "SKILL.md").read_text(encoding="utf-8")


def install(directory: Path, name: str, text: str | None = None) -> Path:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        shipped_text(name) if text is None else text, encoding="utf-8"
    )
    return skill_file


def install_bytes(directory: Path, name: str, data: bytes) -> Path:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_bytes(data)
    return skill_file


def skill_text(name: str) -> str:
    return f"---\nname: {name}\ndescription: A user-authored skill.\n---\n\nBody.\n"


def diverged_text(name: str = SKILL) -> str:
    """The shipped text with its description line replaced."""
    text = shipped_text(name)
    start = text.index("description: ")
    end = text.index("\n", start)
    return text[:start] + "description: User-tuned copy for this project." + text[end:]


def invalid_utf8(name: str = SKILL) -> bytes:
    return skill_text(name).encode("utf-8") + b"\xff\xfe\x00\x01 broken bytes\n"


def build(harness_files: HarnessFilesManager | None = None) -> ShadowInventory:
    config = build_test_vibe_config()
    return build_shadow_inventory(
        lambda: config, harness_files=harness_files or get_harness_files_manager()
    )


def entry_at(inventory: ShadowInventory, skill_file: Path) -> ShadowEntry:
    matches = [
        entry
        for entry in inventory.entries
        if entry.skill_path.parent.resolve() == skill_file.parent.resolve()
    ]
    assert len(matches) == 1
    return matches[0]


def run_cli_captured(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *args: str
) -> tuple[int, str, str]:
    config = build_test_vibe_config()
    manager = get_harness_files_manager()

    async def load() -> tuple[ChartreuxConfigSchema, HarnessFilesManager]:
        return config, manager

    monkeypatch.setattr(skills_command, "_load_config", load)
    with pytest.raises(SystemExit) as exc:
        skills_command.run_skills_cli(list(args))
    captured = capsys.readouterr()
    return int(exc.value.code or 0), captured.out, captured.err


def run_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *args: str
) -> tuple[int, str]:
    code, out, _err = run_cli_captured(monkeypatch, capsys, *args)
    return code, out


def run_cli_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *args: str
) -> tuple[int, dict[str, Any]]:
    code, out = run_cli(monkeypatch, capsys, *args)
    return code, json.loads(out)


def run_doctor(capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    with pytest.raises(SystemExit) as exc:
        doctor.run_doctor_cli(["--json"])
    captured = capsys.readouterr()
    assert captured.err == ""
    return int(exc.value.code or 0), json.loads(captured.out)


def warns(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [check for check in result["checks"] if check["status"] == "warn"]


def skill_warns(result: dict[str, Any]) -> list[dict[str, Any]]:
    # Doctor keeps ONE advisory list: the route section's own warns (O2) are
    # also ``warn`` checks, so this helper names the skill ones explicitly.
    return [
        check
        for check in warns(result)
        if check["name"].startswith(("skill ", "user skill "))
    ]


class TestShadowInventory:
    def test_identical_copy_is_compared_against_shipped(
        self, user_skills: Path
    ) -> None:
        skill_file = install(user_skills, SKILL)
        entry = entry_at(build(), skill_file)
        assert entry.state is ShadowState.IDENTICAL
        assert entry.shipped_path == SHIPPED_SKILLS / SKILL / "SKILL.md"
        assert entry.diff == ""
        assert entry.reason == ""

    def test_diverged_copy_has_shipped_as_the_old_side(self, user_skills: Path) -> None:
        skill_file = install(user_skills, SKILL, diverged_text())
        entry = entry_at(build(), skill_file)
        assert entry.state is ShadowState.DIVERGED
        assert entry.diff.startswith(f"--- shipped:{entry.shipped_path}\n")
        assert f"+++ user:{skill_file}\n" in entry.diff
        removed = next(
            line
            for line in shipped_text().splitlines()
            if line.startswith("description: ")
        )
        assert f"-{removed}\n" in entry.diff
        assert "+description: User-tuned copy for this project.\n" in entry.diff

    def test_shadowed_copy_is_inactive_and_names_the_winner(
        self, user_skills: Path, agents_skills: Path
    ) -> None:
        winner = install(user_skills, SKILL)
        loser = install(agents_skills, SKILL, diverged_text())
        inventory = build()
        inactive = entry_at(inventory, loser)
        assert inactive.state is ShadowState.INACTIVE
        assert inactive.winner_path is not None
        assert inactive.winner_path.resolve() == winner.resolve()
        assert inactive.winner_scope == "global"
        assert inactive.diff == ""
        assert entry_at(inventory, winner).state is ShadowState.IDENTICAL

    def test_builtin_name_copy_is_inactive_without_a_path(
        self, user_skills: Path
    ) -> None:
        skill_file = install(user_skills, "chartreux", skill_text("chartreux"))
        entry = entry_at(build(), skill_file)
        assert entry.state is ShadowState.INACTIVE
        assert entry.winner_path is None
        assert entry.winner_scope == "builtin"
        assert entry.shipped_path is None

    def test_unparsable_copy_is_not_a_shadow(self, user_skills: Path) -> None:
        install(user_skills, "broken", "no frontmatter at all\n")
        assert build().entries == ()

    def test_non_shipping_user_copy_is_not_a_shadow(self, user_skills: Path) -> None:
        install(user_skills, "local-only", skill_text("local-only"))
        assert build().entries == ()

    def test_project_copy_is_never_scanned_as_a_user_copy(
        self, user_skills: Path, tmp_working_directory: Path, config_dir: Path
    ) -> None:
        project_skills = tmp_working_directory / ".chartreux" / "skills"
        project_skills.mkdir(parents=True)
        project_copy = install(project_skills, SKILL, diverged_text())
        user_copy = install(user_skills, SKILL)
        (config_dir / "trusted_folders.toml").write_text(
            f'trusted = ["{tmp_working_directory}"]\n'
        )
        manager = HarnessFilesManager(
            sources=("user", "project"),
            cwd=tmp_working_directory,
            trust_store=TrustedFoldersManager(),
        )
        inventory = build(manager)
        entry = entry_at(inventory, user_copy)
        assert entry.state is ShadowState.INACTIVE
        assert entry.winner_path is not None
        assert entry.winner_path.resolve() == project_copy.resolve()
        assert entry.winner_scope == "project"
        scanned = {
            skill_entry.skill_path.resolve() for skill_entry in inventory.entries
        }
        assert project_copy.resolve() not in scanned
        assert all(
            skill_entry.user_dir
            in (user_skills, config_dir.parent / ".agents" / "skills")
            for skill_entry in inventory.entries
        )

    def test_entries_are_sorted_by_name_then_path(
        self, user_skills: Path, agents_skills: Path
    ) -> None:
        install(user_skills, "sub-researcher")
        install(user_skills, SKILL, diverged_text())
        install(agents_skills, SKILL)
        names = [(entry.name, str(entry.skill_path)) for entry in build().entries]
        assert names == sorted(names)

    def test_inventory_does_not_change_loader_selection(
        self, user_skills: Path
    ) -> None:
        skill_file = install(user_skills, SKILL, diverged_text())
        config = build_test_vibe_config()
        loader = skills_manager.SkillManager(lambda: config)
        selected = loader.get_skill(SKILL)
        assert selected is not None
        assert selected.skill_path is not None
        assert selected.skill_path.resolve() == skill_file.resolve()
        inventory = build_shadow_inventory(
            lambda: config, harness_files=get_harness_files_manager()
        )
        assert entry_at(inventory, skill_file).state is ShadowState.DIVERGED
        assert loader.get_skill(SKILL) is selected


MULTI_HUNK_SHIPPED = (
    "---\n"
    "name: demo-skill\n"
    "description: Shipped description.\n"
    "---\n"
    "\n"
    "# Demo\n"
    "\n"
    "line one\n"
    "line two\n"
    "line three\n"
    "line four\n"
    "line five\n"
    "line six\n"
    "line seven\n"
    "line eight\n"
    "line nine\n"
    "line ten\n"
    "line eleven\n"
)

MULTI_HUNK_USER = MULTI_HUNK_SHIPPED.replace(
    "description: Shipped description.", "description: User description."
).replace("line ten", "line TEN")


class TestSkillsDiffCommand:
    def test_identical_copy_exits_zero_with_removal_wording(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        user_skills: Path,
    ) -> None:
        install(user_skills, SKILL)
        code, out = run_cli(monkeypatch, capsys, "diff")
        assert code == 0
        assert f"identical: {SKILL}" in out
        assert IDENTICAL_WORDING in out

    def test_diverged_copy_exits_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        user_skills: Path,
    ) -> None:
        install(user_skills, SKILL, diverged_text())
        code, out = run_cli(monkeypatch, capsys, "diff")
        assert code == 1
        assert f"diverged: {SKILL}" in out
        assert "--- shipped:" in out
        assert "+++ user:" in out
        assert "+description: User-tuned copy for this project." in out

    def test_inactive_copy_is_reported_with_its_winner(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        user_skills: Path,
        agents_skills: Path,
    ) -> None:
        winner = install(user_skills, SKILL)
        install(agents_skills, SKILL)
        code, out = run_cli(monkeypatch, capsys, "diff")
        assert code == 0
        assert f"inactive: {SKILL}" in out
        assert str(winner) in out
        assert "(global)" in out

    def test_invalid_utf8_is_an_operational_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        user_skills: Path,
    ) -> None:
        install_bytes(user_skills, SKILL, invalid_utf8())
        code, out = run_cli(monkeypatch, capsys, "diff")
        assert code == 2
        assert f"unverified: {SKILL}" in out
        assert "UTF-8" in out

    def test_errors_take_precedence_over_differences(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        user_skills: Path,
    ) -> None:
        install(user_skills, SKILL, diverged_text())
        install_bytes(user_skills, "sub-researcher", invalid_utf8("sub-researcher"))
        code, result = run_cli_json(monkeypatch, capsys, "diff", "--json")
        assert code == 2
        assert result["exit_code"] == 2
        states = {entry["state"] for entry in result["entries"]}
        assert states == {"diverged", "unverified"}

    def test_name_filter_reports_one_skill(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        user_skills: Path,
    ) -> None:
        install(user_skills, SKILL, diverged_text())
        install(user_skills, "sub-researcher")
        code, result = run_cli_json(monkeypatch, capsys, "diff", SKILL, "--json")
        assert code == 1
        assert [entry["name"] for entry in result["entries"]] == [SKILL]

    def test_unknown_name_exits_two(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        user_skills: Path,
    ) -> None:
        install(user_skills, SKILL)
        code, out, err = run_cli_captured(monkeypatch, capsys, "diff", "does-not-exist")
        assert code == 2
        assert out == ""
        assert "unknown skill" in err

    def test_json_key_set_is_pinned(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        user_skills: Path,
    ) -> None:
        install(user_skills, SKILL)
        code, result = run_cli_json(monkeypatch, capsys, "diff", "--json")
        assert code == 0
        assert set(result) == {"entries", "exit_code"}
        assert result["exit_code"] == 0
        assert len(result["entries"]) == 1
        assert set(result["entries"][0]) == ENTRY_KEYS

    def test_multi_hunk_diff_is_byte_stable_and_ordered(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        user_skills: Path,
    ) -> None:
        shipped_dir = tmp_path / "shipped-skills"
        (shipped_dir / "demo-skill").mkdir(parents=True)
        (shipped_dir / "demo-skill" / "SKILL.md").write_text(MULTI_HUNK_SHIPPED)
        monkeypatch.setattr(skills_manager, "SHIPPED_SKILLS_DIR", shipped_dir)
        skill_file = install(user_skills, "demo-skill", MULTI_HUNK_USER)
        shipped_path = shipped_dir / "demo-skill" / "SKILL.md"
        expected = (
            f"diverged: demo-skill\n"
            f"  shipped: {shipped_path}\n"
            f"  user:    {skill_file}\n"
            f"--- shipped:{shipped_path}\n"
            f"+++ user:{skill_file}\n"
            "@@ -1,6 +1,6 @@\n"
            " ---\n"
            " name: demo-skill\n"
            "-description: Shipped description.\n"
            "+description: User description.\n"
            " ---\n"
            " \n"
            " # Demo\n"
            "@@ -14,5 +14,5 @@\n"
            " line seven\n"
            " line eight\n"
            " line nine\n"
            "-line ten\n"
            "+line TEN\n"
            " line eleven\n"
        )
        code, first = run_cli(monkeypatch, capsys, "diff", "demo-skill")
        assert code == 1
        assert first.rstrip("\n") == expected.rstrip("\n")
        _, second = run_cli(monkeypatch, capsys, "diff", "demo-skill")
        assert second == first

    def test_no_user_skills_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out = run_cli(monkeypatch, capsys, "diff")
        assert code == 0
        assert "no user skill directories exist yet" in out

    def test_real_config_load_reaches_the_inventory(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        user_skills: Path,
    ) -> None:
        install(user_skills, SKILL)
        monkeypatch.setattr(
            "chartreux.core.config.chartreux_schema.load_dotenv_values", lambda: None
        )
        with pytest.raises(SystemExit) as exc:
            skills_command.run_skills_cli(["diff", "--json"])
        assert exc.value.code == 0
        result = json.loads(capsys.readouterr().out)
        assert [entry["name"] for entry in result["entries"]] == [SKILL]


class TestEntrypointDispatch:
    def test_entrypoint_dispatches_the_skills_subcommand(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        user_skills: Path,
    ) -> None:
        from chartreux.cli import entrypoint

        install(user_skills, SKILL)
        monkeypatch.setattr("sys.argv", ["chartreux", "skills", "diff", "--json"])
        monkeypatch.setattr(
            "chartreux.core.config.chartreux_schema.load_dotenv_values", lambda: None
        )
        with pytest.raises(SystemExit) as exc:
            entrypoint.main()
        assert exc.value.code == 0
        result = json.loads(capsys.readouterr().out)
        assert [entry["name"] for entry in result["entries"]] == [SKILL]


class TestDoctorShadowChecks:
    def test_identical_copy_warns_with_removal_wording(
        self, capsys: pytest.CaptureFixture[str], user_skills: Path
    ) -> None:
        install(user_skills, SKILL)
        code, result = run_doctor(capsys)
        warnings = skill_warns(result)
        assert code == 0
        assert len(warnings) == 1
        assert warnings[0]["name"] == f"user skill {SKILL}"
        assert IDENTICAL_WORDING in warnings[0]["reason"]

    def test_diverged_copy_warns_with_the_command(
        self, capsys: pytest.CaptureFixture[str], user_skills: Path
    ) -> None:
        install(user_skills, SKILL, diverged_text())
        code, result = run_doctor(capsys)
        warnings = skill_warns(result)
        assert code == 0
        assert len(warnings) == 1
        assert "chartreux skills diff" in warnings[0]["reason"]

    def test_inactive_copy_warns_with_the_winner(
        self, capsys: pytest.CaptureFixture[str], user_skills: Path, agents_skills: Path
    ) -> None:
        winner = install(user_skills, SKILL)
        install(agents_skills, SKILL)
        code, result = run_doctor(capsys)
        warnings = skill_warns(result)
        assert code == 0
        assert len(warnings) == 2
        inactive = [check for check in warnings if "inactive" in check["reason"]]
        assert len(inactive) == 1
        assert str(winner) in inactive[0]["reason"]
        assert "(global)" in inactive[0]["reason"]

    def test_unverified_copy_warns_on_comparison_failure(
        self, capsys: pytest.CaptureFixture[str], user_skills: Path
    ) -> None:
        install_bytes(user_skills, SKILL, invalid_utf8())
        code, result = run_doctor(capsys)
        warnings = skill_warns(result)
        assert code == 0
        assert len(warnings) == 1
        assert warnings[0]["reason"].startswith("unverified:")

    def test_no_shadowed_copy_is_a_pass(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, result = run_doctor(capsys)
        checks = {check["name"]: check for check in result["checks"]}
        assert code == 0
        assert checks["user skills"]["status"] == "pass"
        assert skill_warns(result) == []


class TestSharedInventory:
    def test_both_surfaces_render_one_shared_builder(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        user_skills: Path,
    ) -> None:
        install(user_skills, SKILL, diverged_text())
        real = skills_manager.build_shadow_inventory
        rendered: list[ShadowInventory] = []

        def spy(*args: Any, **kwargs: Any) -> ShadowInventory:
            inventory = real(*args, **kwargs)
            rendered.append(inventory)
            return inventory

        monkeypatch.setattr(skills_manager, "build_shadow_inventory", spy)
        code, result = run_doctor(capsys)
        assert code == 0
        assert len(rendered) == 1
        assert warns(result)[0]["name"] == f"user skill {SKILL}"
        rendered.clear()
        code, _out = run_cli(monkeypatch, capsys, "diff")
        assert code == 1
        assert len(rendered) == 1
