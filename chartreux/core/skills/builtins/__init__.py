from __future__ import annotations

from chartreux.core.skills.builtins.chartreux import SKILL as CHARTREUX_SKILL
from chartreux.core.skills.builtins.skill_creator import SKILL as SKILL_CREATOR_SKILL
from chartreux.core.skills.models import SkillInfo

BUILTIN_SKILLS: dict[str, SkillInfo] = {
    skill.name: skill for skill in [CHARTREUX_SKILL, SKILL_CREATOR_SKILL]
}
