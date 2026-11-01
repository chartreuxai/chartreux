"""K2 acceptance: typed skill-load failures on the SINGLE skill inventory."""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.config.harness_files import get_harness_files_manager
from chartreux.core.skills import manager as skills_manager
from chartreux.core.skills.manager import (
    SkillManager,
    build_shadow_inventory,
    sanitized_failure_diagnostic,
)
from chartreux.core.skills.models import (
    SkillConfigIssue,
    SkillMetadata,
    SkillScope,
    SkillSource,
)
from chartreux.core.skills.parser import SkillParseError
from tests.conftest import build_test_vibe_config

SECRET = "super-secret-frontmatter-value"

# A parseable mapping that fails metadata validation: the missing description
# and the invalid name are both reportable WITHOUT echoing the value.
BROKEN_FRONTMATTER = f"---\nname: {SECRET}\n---\n\nNo description field.\n"
# YAML errors quote the offending source in their message.
BROKEN_YAML = f"---\nname: [unterminated\nsecret: {SECRET}\n---\n\nBody.\n"


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


def build_manager() -> SkillManager:
    return SkillManager(lambda: build_test_vibe_config())


class TestLoadFailureRecords:
    def test_broken_skill_records_one_typed_failure(self, user_skills: Path) -> None:
        skill_file = install(user_skills, "broken-skill", BROKEN_FRONTMATTER)
        (failure,) = build_manager().load_failures
        assert failure.path.resolve() == skill_file.resolve()
        assert failure.scope is SkillScope.GLOBAL
        assert failure.source is SkillSource.LOCAL
        assert failure.origin == "global"
        assert failure.diagnostic.startswith("invalid skill metadata")

    def test_record_carries_no_parsed_name(self, user_skills: Path) -> None:
        install(user_skills, "broken-skill", BROKEN_FRONTMATTER)
        (failure,) = build_manager().load_failures
        # A file that fails to load may have no valid name; surfaces render
        # the path, so the record must not require one.
        assert not hasattr(failure, "name")

    def test_broken_skill_never_blocks_startup(self, user_skills: Path) -> None:
        install(user_skills, "broken-skill", BROKEN_FRONTMATTER)
        manager = build_manager()
        assert manager.get_skill("broken-skill") is None
        # Discovery completes: builtin and shipped skills still load.
        assert "chartreux" in manager.available_skills
        assert manager.custom_skills_count == 0

    def test_repaired_file_loads_on_the_next_discovery(self, user_skills: Path) -> None:
        skill_file = install(user_skills, "broken-skill", BROKEN_FRONTMATTER)
        assert build_manager().load_failures
        skill_file.write_text(
            "---\nname: broken-skill\ndescription: Repaired.\n---\n\nBody.\n",
            encoding="utf-8",
        )
        repaired = build_manager()
        assert repaired.load_failures == ()
        assert repaired.get_skill("broken-skill") is not None


class TestSanitizedDiagnostics:
    def test_frontmatter_values_never_surface(self, user_skills: Path) -> None:
        install(user_skills, "broken-skill", BROKEN_FRONTMATTER)
        manager = build_manager()
        (failure,) = manager.load_failures
        (issue,) = manager.config_issues
        assert SECRET not in failure.diagnostic
        assert SECRET not in issue.message
        assert issue.file == failure.path
        assert issue.message == f"Failed to load: {failure.diagnostic}"

    def test_yaml_errors_drop_the_source_snippet(self, user_skills: Path) -> None:
        install(user_skills, "broken-skill", BROKEN_YAML)
        (failure,) = build_manager().load_failures
        assert failure.diagnostic == "Invalid YAML frontmatter"
        assert SECRET not in failure.diagnostic

    def test_validation_errors_keep_only_fields_and_kinds(self) -> None:
        value = f"{SECRET} with spaces"
        with pytest.raises(Exception) as excinfo:
            SkillMetadata.model_validate({"name": value})
        diagnostic = sanitized_failure_diagnostic(excinfo.value)
        assert diagnostic == (
            "invalid skill metadata: name: string_pattern_mismatch; description: missing"
        )
        assert SECRET not in diagnostic

    def test_read_failures_carry_no_os_detail(self) -> None:
        diagnostic = sanitized_failure_diagnostic(
            SkillParseError("Cannot read file: [Errno 13] Permission denied: '/etc/x'")
        )
        assert diagnostic == "Cannot read file"

    def test_diagnostic_is_bounded(self) -> None:
        class Detail(Exception):
            pass

        diagnostic = sanitized_failure_diagnostic(
            SkillParseError("x" * (skills_manager._MAX_DIAGNOSTIC_CHARS * 2))
        )
        assert len(diagnostic) == skills_manager._MAX_DIAGNOSTIC_CHARS
        assert diagnostic.endswith("...")
        assert sanitized_failure_diagnostic(Detail()).startswith("unexpected Detail")


class TestSingleFailureSource:
    def test_config_issues_project_the_failure_records(self, user_skills: Path) -> None:
        skill_file = install(user_skills, "broken-skill", BROKEN_FRONTMATTER)
        manager = build_manager()
        (record,) = manager.load_failures
        assert manager.config_issues == (
            SkillConfigIssue(
                file=skill_file, message=f"Failed to load: {record.diagnostic}"
            ),
        )
        # No parallel list survives discovery: the projection is the only
        # reader, so emptying the single record source empties both surfaces.
        assert not hasattr(manager, "_config_issues")

    def test_emptying_the_records_empties_the_projection(
        self, monkeypatch: pytest.MonkeyPatch, user_skills: Path
    ) -> None:
        install(user_skills, "broken-skill", BROKEN_FRONTMATTER)
        manager = build_manager()
        assert manager.config_issues
        monkeypatch.setattr(SkillManager, "load_failures", property(lambda self: ()))
        assert manager.config_issues == ()


class TestInventorySnapshot:
    def test_inventory_records_come_from_one_discovery_pass(
        self, monkeypatch: pytest.MonkeyPatch, user_skills: Path
    ) -> None:
        install(user_skills, "broken-skill", BROKEN_FRONTMATTER)
        real = skills_manager.SkillManager
        builds: list[SkillManager] = []

        def spy(*args: object, **kwargs: object) -> SkillManager:
            manager = real(*args, **kwargs)  # type: ignore[arg-type]
            builds.append(manager)
            return manager

        monkeypatch.setattr(skills_manager, "SkillManager", spy)
        config = build_test_vibe_config()
        inventory = build_shadow_inventory(
            lambda: config, harness_files=get_harness_files_manager()
        )
        # One inventory, one discovery pass: no second scan builds the records.
        assert len(builds) == 1
        assert inventory.load_failures == builds[0].load_failures
        assert len(inventory.load_failures) == 1
        assert inventory.entries == ()

    def test_inventory_records_match_a_plain_discovery(self, user_skills: Path) -> None:
        install(user_skills, "broken-skill", BROKEN_FRONTMATTER)
        config = build_test_vibe_config()
        inventory = build_shadow_inventory(
            lambda: config, harness_files=get_harness_files_manager()
        )
        assert inventory.load_failures == build_manager().load_failures
