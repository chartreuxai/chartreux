"""v0.7.6 P2 chunk 2f: the notification-first guidance rewrite (E0 inventory).

Every tiled guidance site in the E0 inventory stops teaching "launch, then
collect with wait_for_agent" and instead teaches
launch -> completion notification -> targeted fetch (digest first, full on
demand), naming both delivery modes wherever fetch teaching occurs.

Two inventory sites are pinned by tests outside this chunk's owned file list
and are therefore NOT rewritten here; ``TestPinnedCollisionSites`` documents
them and asserts the pins stay intact:

- the launch-ack text in ``app_server/_subagent_execution.py`` is pinned by
  exact response equality in ``tests/app_server/test_subagents_launch.py``
  and by prefix in ``tests/app_server/test_subagent_management_surface.py``;
- the still-running fallback message in ``get_agent_result.py``'s legacy
  (test-double) branch is pinned by ``tests/tools/test_agent_management.py``;
  only the P2d delivery-path branch was rewritten;
- the fixture-mirror line 19 old claim renders from
  ``core/dispatch/renderer.py``'s ``_COMPOSITION_REUSE`` block, which is
  outside this chunk's owned files, so it was left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from chartreux.core.skills.builtins import BUILTIN_SKILLS

ROOT = Path(__file__).parents[2]

#: The notification-first marker phrase every rewritten site must carry.
NOTIFICATION_FIRST = "completion notification"

#: Both delivery modes must be named at every fetch-teaching site.
MODE_DIGEST = 'mode="digest"'
MODE_FULL = 'mode="full"'


@dataclass(frozen=True)
class Site:
    """One tiled site of the E0 guidance-string inventory."""

    site_id: str
    relpath: str
    old_claims: tuple[str, ...]
    required: tuple[str, ...]
    skill_prompt: bool = False


def _read(site: Site) -> str:
    if site.skill_prompt:
        return BUILTIN_SKILLS["chartreux"].prompt
    return (ROOT / site.relpath).read_text()


TASK_MD_OLD = (
    "so launch, then collect the result with `wait_for_agent`",
    "the result stays retrievable with `get_agent_result` or `wait_for_agent`",
    "`get_agent_result(agent_id, run_id)` to retrieve a completed result "
    "without waiting",
)
TASK_MD_REQUIRED = (NOTIFICATION_FIRST, MODE_DIGEST, MODE_FULL)

INVENTORY: tuple[Site, ...] = (
    Site(
        "task.md:9,:11,:31",
        "chartreux/core/tools/builtins/prompts/task.md",
        TASK_MD_OLD,
        TASK_MD_REQUIRED,
    ),
    Site(
        "fixture standalone/task.md:9,:11,:68",
        "tests/fixtures/dispatch/standalone/task.md",
        TASK_MD_OLD,
        TASK_MD_REQUIRED,
    ),
    Site(
        "wait_for_agent.md:1",
        "chartreux/core/tools/builtins/prompts/wait_for_agent.md",
        ("for a specific invocation, and optionally a positive `timeout`",),
        (NOTIFICATION_FIRST, '"digest"', '"full"'),
    ),
    Site(
        "cancel_agent.md:3",
        "chartreux/core/tools/builtins/prompts/cancel_agent.md",
        (
            "Use `wait_for_agent` or `get_agent_result` to retrieve the "
            "terminal result after cleanup",
        ),
        (NOTIFICATION_FIRST, MODE_DIGEST, MODE_FULL),
    ),
    Site(
        "check_agents.md:1",
        "chartreux/core/tools/builtins/prompts/check_agents.md",
        (
            "Collect one with `get_agent_result(agent_id, run_id=...)` or "
            "`wait_for_agent`",
        ),
        (NOTIFICATION_FIRST, MODE_DIGEST, MODE_FULL),
    ),
    Site(
        "get_agent_result.py delivery-path pending message",
        "chartreux/core/tools/builtins/get_agent_result.py",
        (),
        ("a completion notification arrives", 'mode="digest"', 'mode="full"'),
    ),
    Site(
        "chartreux.py:728 (builtin skill prompt)",
        "chartreux/core/skills/builtins/chartreux.py",
        ("then wait for it with `wait_for_agent`", "wait if its result is needed"),
        (NOTIFICATION_FIRST, MODE_DIGEST, MODE_FULL),
        skill_prompt=True,
    ),
    Site(
        "main-plan SKILL.md:27",
        "chartreux/skills/main-plan/SKILL.md",
        ("collect the advisor's result with `wait_for_agent`",),
        (NOTIFICATION_FIRST, MODE_DIGEST, MODE_FULL),
    ),
    Site(
        "main-design SKILL.md:27",
        "chartreux/skills/main-design/SKILL.md",
        ("collect the advisor's result with `wait_for_agent`",),
        (NOTIFICATION_FIRST, MODE_DIGEST, MODE_FULL),
    ),
    Site(
        "subagents.py:279-291,:711-718",
        "chartreux/core/subagents.py",
        (
            "re-invoke wait_for_agent in a later turn. The variants are mutually",
            "to keep waiting, or call check_agents to inspect progress.",
        ),
        (NOTIFICATION_FIRST, 'mode="digest"', 'mode="full"'),
    ),
)


class TestGuidanceInventory:
    """Per-site content assertions: old claim absent, new language present."""

    @pytest.mark.parametrize("site", INVENTORY, ids=[s.site_id for s in INVENTORY])
    def test_old_claim_absent_and_notification_first_present(self, site: Site) -> None:
        text = _read(site)
        for old in site.old_claims:
            assert old not in text, f"{site.site_id}: old claim still present: {old!r}"
        for required in site.required:
            assert required in text, (
                f"{site.site_id}: required language missing: {required!r}"
            )

    @pytest.mark.parametrize("site", INVENTORY, ids=[s.site_id for s in INVENTORY])
    def test_both_delivery_modes_named(self, site: Site) -> None:
        text = _read(site)
        assert "digest" in text, f"{site.site_id}: digest mode not named"
        assert "full" in text, f"{site.site_id}: full mode not named"

    def test_inventory_covers_every_e0_entry(self) -> None:
        # The test enumerates the full E0 guidance-string inventory tiled by
        # chunk 2f: task.md (prompt + fixture mirror), the three lifecycle
        # tool prompts, get_agent_result.py, the builtin chartreux skill,
        # both shipped advisor SKILL.md files, and core/subagents.py. The
        # remaining inventory entry (the launch-ack text) is a pinned
        # collision documented in TestPinnedCollisionSites.
        assert len(INVENTORY) == 10
        relpaths = {s.relpath for s in INVENTORY}
        assert relpaths == {
            "chartreux/core/tools/builtins/prompts/task.md",
            "tests/fixtures/dispatch/standalone/task.md",
            "chartreux/core/tools/builtins/prompts/wait_for_agent.md",
            "chartreux/core/tools/builtins/prompts/cancel_agent.md",
            "chartreux/core/tools/builtins/prompts/check_agents.md",
            "chartreux/core/tools/builtins/get_agent_result.py",
            "chartreux/core/skills/builtins/chartreux.py",
            "chartreux/skills/main-plan/SKILL.md",
            "chartreux/skills/main-design/SKILL.md",
            "chartreux/core/subagents.py",
        }


class TestConsistentLanguage:
    """The same lifecycle vocabulary is used at every rewritten site."""

    @pytest.mark.parametrize("site", INVENTORY, ids=[s.site_id for s in INVENTORY])
    def test_completion_notification_phrase(self, site: Site) -> None:
        assert NOTIFICATION_FIRST in _read(site), site.site_id

    def test_fixture_mirror_matches_the_skeleton(self) -> None:
        skeleton = (ROOT / INVENTORY[0].relpath).read_text()
        fixture = (ROOT / INVENTORY[1].relpath).read_text()
        for required in TASK_MD_REQUIRED:
            assert required in skeleton
            assert required in fixture


class TestGetAgentResultDeliveryBranch:
    """The P2d pending branch teaches notification-first; the legacy branch
    keeps its pinned message (test-double fallback).
    """

    def test_pending_branch_is_notification_first(self) -> None:
        source = (
            ROOT / "chartreux/core/tools/builtins/get_agent_result.py"
        ).read_text()
        start = source.index('if status == "pending":')
        end = source.index('if status in {"refused", "nothing_left"}:')
        branch = source[start:end]
        assert NOTIFICATION_FIRST in branch
        assert 'mode="digest"' in branch
        assert 'mode="full"' in branch
        assert "use wait_for_agent or try again later" not in branch

    def test_legacy_branch_keeps_the_pinned_message_once(self) -> None:
        # tests/tools/test_agent_management.py:513-515 pins the legacy
        # (manager-without-_run_results) pending message verbatim.
        source = (
            ROOT / "chartreux/core/tools/builtins/get_agent_result.py"
        ).read_text()
        assert source.count("use wait_for_agent or try again later") == 1


class TestPinnedCollisionSites:
    """Inventory sites whose rewrite is blocked by pins outside this chunk's
    owned files — asserted intact so the collision is visible, not silent.
    """

    def test_launch_ack_text_still_carries_the_pinned_response(self) -> None:
        # tests/app_server/test_subagents_launch.py:768-777 pins the full
        # acknowledgment response by equality; test_subagent_management_
        # surface.py:480 pins its prefix. Rewriting it would break those
        # unowned tests, so the site is left as-is and reported.
        source = (ROOT / "chartreux/app_server/_subagent_execution.py").read_text()
        assert "launched and running; use check_agents to monitor, " in source
        assert "get_agent_result or wait_for_agent to retrieve the " in source

    def test_renderer_reuse_block_outside_owned_files(self) -> None:
        # The fixture-mirror line 19 claim renders from core/dispatch/renderer.py's
        # _COMPOSITION_REUSE block, which this chunk did not own; v0.7.8 package
        # 13 reconciled that clause to notification-first and regenerated the
        # fixture, so the pin follows the rendered clause in both places.
        renderer = (ROOT / "chartreux/core/dispatch/renderer.py").read_text()
        fixture = (ROOT / "tests/fixtures/dispatch/standalone/task.md").read_text()
        assert "collect its result on the completion notification" in renderer
        assert "collect its result on the completion notification" in fixture


class TestPinnedPhrasesPreserved:
    """Phrases pinned by tests/skills/test_builtin_sync.py and
    tests/cli/test_commands.py around the rewritten regions stay green.
    """

    def test_check_agents_prompt_pins_survive(self) -> None:
        prompt = (
            ROOT / "chartreux/core/tools/builtins/prompts/check_agents.md"
        ).read_text()
        assert (
            "Availability is `running`, `finalizing`, `idle`, `suspended`, "
            "`restoring`, `evicted`, or `closed`." in prompt
        )
        assert "cannot yet be reused" in prompt
        assert "A `closed` agent completed without retention" in prompt
        assert "An `evicted` agent was retained but its runtime was removed" in prompt
        assert (
            "remain available through `get_agent_result` until result expiry" in prompt
        )

    def test_chartreux_skill_pins_survive(self) -> None:
        prompt = BUILTIN_SKILLS["chartreux"].prompt
        for documented in (
            "[subagents]",
            "task_summary",
            "check_agents",
            "get_agent_result",
            "wait_for_agent",
            "release_agent",
            "ttl_remaining_seconds",
            "still_running",
            "1500",
            "display-only",
        ):
            assert documented in prompt, documented
        assert prompt.count("Reuse an `idle` agent") == 1
        assert prompt.count("Persona is immutable") == 1
        assert prompt.count("A `running` agent is busy") == 1

    def test_task_prompt_lifecycle_pins_survive(self) -> None:
        # tests/core/dispatch/test_rendering.py pins these in the rendered
        # task description; tests/test_system_prompt.py pins the cancel
        # paragraph prefix.
        prompt = (ROOT / "chartreux/core/tools/builtins/prompts/task.md").read_text()
        assert "Launch in the background even when the result is a dependency" in prompt
        assert "Retention is opt-in." in prompt
        assert "for a final continuation that closes afterward" in prompt
        assert "Use `cancel_agent(agent_id, run_id)`" in prompt
