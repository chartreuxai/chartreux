"""K2 acceptance: a broken SKILL.md warns in the TUI and doctor, /reload clears it."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from chartreux.cli import doctor_command as doctor
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.core.skills.manager import SkillManager
from tests.conftest import (
    build_test_chartreux_app,
    build_test_vibe_config,
    stub_config_reload,
)

BROKEN_SKILL_TEXT = "---\nname: broken-skill\n---\n\nNo description field.\n"
REPAIRED_SKILL_TEXT = "---\nname: broken-skill\ndescription: Repaired.\n---\n\nBody.\n"
DOCTOR_REMEDIATION = "fix the file and rerun chartreux doctor"


@pytest.fixture
def user_skills(config_dir: Path) -> Path:
    """The first USER skill directory (~/.chartreux/skills)."""
    path = config_dir / "skills"
    path.mkdir()
    return path


def install(directory: Path, name: str, text: str) -> Path:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(text, encoding="utf-8")
    return skill_file


def run_doctor(capsys: pytest.CaptureFixture[str]) -> tuple[int, dict]:
    with pytest.raises(SystemExit) as exc:
        doctor.run_doctor_cli(["--json"])
    captured = capsys.readouterr()
    assert captured.err == ""
    return int(exc.value.code or 0), json.loads(captured.out)


def skill_warns(result: dict) -> list[dict]:
    # Doctor keeps ONE advisory list: the route section's own warns (O2) are
    # also ``warn`` checks, so this helper names the skill ones explicitly.
    return [
        check
        for check in result["checks"]
        if check["status"] == "warn"
        and check["name"].startswith(("skill ", "user skill "))
    ]


def surfaced(app: ChartreuxApp, skill_file: Path) -> list[str]:
    # The shared issue list also carries route rows: name this skill's file.
    issues = app._recovery_issues
    return [text for _severity, text in issues.values() if str(skill_file) in text]


def test_broken_skill_warns_then_reload_restores_and_clears(
    user_skills: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The e2e acceptance: TUI warn + doctor warn, then /reload clears both."""
    skill_file = install(user_skills, "broken-skill", BROKEN_SKILL_TEXT)

    code, result = run_doctor(capsys)
    assert code == 0
    (warning,) = skill_warns(result)
    assert str(skill_file) in warning["reason"]
    assert DOCTOR_REMEDIATION in warning["reason"]

    async def repair_and_reload() -> None:
        config = build_test_vibe_config()
        app = build_test_chartreux_app(config=config)
        async with app.run_test() as pilot:
            runtime = app.app_server.resources.runtime
            await runtime.wait_until_ready()
            app._show_config_issues()
            await pilot.pause(0.05)

            # The broken skill never blocks startup; the TUI warns with
            # /reload remediation and the sanitized diagnostic.
            assert runtime.get_skill("broken-skill") is None
            (message,) = surfaced(app, skill_file)
            assert (
                "Failed to load: invalid skill metadata: description: missing"
                in message
            )
            assert "/reload" in message
            assert app.query_one("#recovery-notice").display

            # Repair the file, then run the real /reload path: discovery
            # reruns, the skill returns, and the warning clears.
            skill_file.write_text(REPAIRED_SKILL_TEXT, encoding="utf-8")
            stub_config_reload(monkeypatch, config)
            await app._reload_config()
            await pilot.pause(0.1)

            assert surfaced(app, skill_file) == []
            assert [
                issue for issue in runtime.issues if str(skill_file) in issue.file
            ] == []
            assert runtime.get_skill("broken-skill") is not None

    asyncio.run(repair_and_reload())

    code, result = run_doctor(capsys)
    assert code == 0
    assert skill_warns(result) == []


@pytest.mark.asyncio
async def test_tui_issue_list_projects_the_single_failure_record_source(
    user_skills: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill_file = install(user_skills, "broken-skill", BROKEN_SKILL_TEXT)
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        runtime = app.app_server.resources.runtime
        await runtime.wait_until_ready()
        app._show_config_issues()
        await pilot.pause(0.05)
        assert len(surfaced(app, skill_file)) == 1

        # The TUI list has no second source: emptying the typed records
        # empties the projected issues and the rendered warning together.
        monkeypatch.setattr(SkillManager, "load_failures", property(lambda self: ()))
        await runtime.refresh()
        app._show_config_issues()
        assert surfaced(app, skill_file) == []
