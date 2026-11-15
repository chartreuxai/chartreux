"""Notification-first subagent-collection guidance (v0.7.6 inventory, v0.7.9 rule).

Every tiled guidance site in the E0 inventory stops teaching "launch, then
collect with wait_for_agent" and instead teaches
launch -> completion notification -> targeted fetch (digest first, full on
demand), naming both delivery modes wherever fetch teaching occurs.

v0.7.9 point 4 reconditions the same inventory to the decision rule: three wait
conditions, the uncertain-case yield, and the anti-invention clause. The three
sites v0.7.6 recorded as pinned collisions — the launch acknowledgment in
``app_server/_subagent_execution.py``, the legacy pending fallback in
``get_agent_result.py``, and the ``_COMPOSITION_REUSE`` renderer clause — are
rewritten here together with the pins that blocked them, so no site asserts
the old co-equal-wait guidance anymore.
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

#: The anti-invention clause of the shipped decision rule.
ANTI_INVENTION = (
    "Do not invent independent work or dispatch unnecessary tasks to fill the "
    "wait; if there is nothing real to do, end your turn."
)


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
    "or block on the run with `wait_for_agent`",
)
TASK_MD_REQUIRED = (NOTIFICATION_FIRST, MODE_DIGEST, MODE_FULL)

#: The three wait conditions, the uncertain-case yield, and the
#: anti-invention clause, as this site words them.
TASK_RULE = (
    "even if your next substantive action depends on it",
    "Use `wait_for_agent` only if the result is a hard dependency of your next "
    "action, no independent work can proceed, and that action must occur in "
    "this turn rather than after notification",
    "If any of those conditions is uncertain, end your turn.",
    ANTI_INVENTION,
)
WFA_RULE = (
    "Use this blocking tool only if the result is a hard dependency of your "
    "next action, no independent work can proceed, and that action must occur "
    "in this turn rather than after notification",
    "If uncertain, end your turn.",
    ANTI_INVENTION,
    "never re-invoke the wait in the same turn",
)
CANCEL_RULE = (
    "only if the result is a hard dependency of your next action, no "
    "independent work can proceed, and that action must occur in this turn "
    "rather than after notification; if any of those conditions is uncertain, "
    "end your turn",
    "Acceptance of the request is not proof of completion.",
)
CHECK_RULE = ("not a blocking wait", "uncollected_run_ids")
#: Python source wraps literals across physical lines, so source-file checks
#: use fragments that each sit inside one literal; the rendered/markdown sites
#: use whole sentences above.
GAR_RULE = (
    "Agent is still running; a completion notification arrives ",
    'mode="digest" (default) or mode="full", or collect a ',
    "uncollected_run_ids. Use wait_for_agent (same modes) only ",
    "if the result is a hard dependency of your next action, no ",
    "independent work can proceed, and that action must occur in ",
    "this turn rather than after notification; if uncertain, ",
    "end your turn.",
)
LEGACY_GAR_RULE = (
    "Agent is still running; fetch the result when the ",
    "completion notification arrives, or collect it by ",
    "run_id from check_agents' uncollected_run_ids. Use ",
    "wait_for_agent only if the result is a hard dependency ",
    "of your next action, no independent work can proceed, ",
    "and that action must occur in this turn rather than ",
    "after notification; if uncertain, end your turn.",
)
ACK_RULE = (
    "launched and running. Do independent work or end your turn; on the ",
    "completion notification, fetch the result with get_agent_result. ",
    "Use wait_for_agent only if the result is a hard dependency of your ",
    "next action, no independent work can proceed, and that action must ",
    "occur in this turn rather than after notification. If uncertain, ",
    "end your turn. Do not invent work to fill the wait; if there is ",
    "nothing real to do, end your turn. Use check_agents to inspect ",
)
BUILTIN_SKILL_RULE = (
    "Use `wait_for_agent` only if the result is a hard dependency of your next "
    "action, no independent work can proceed, and that action must occur in "
    "this turn rather than after notification; if any of those conditions is "
    "uncertain, end your turn.",
    ANTI_INVENTION,
)
ADVISOR_SKILL_RULE = (
    "Use `wait_for_agent` only if the result is a hard dependency of your next "
    "action, no independent work can proceed, and synthesis must occur in this "
    "turn rather than after notification. If uncertain, end your turn.",
    ANTI_INVENTION,
)
SUBAGENTS_RULE = (
    "Never re-invoke ",
    "wait_for_agent in the same turn. In a later turn, re-wait only if ",
    "the result is a hard dependency of your next action, no independent ",
    "work can proceed, and that action must occur in that turn rather ",
    "than after notification; if uncertain, end your turn. Use ",
    "between turns. Prefer the completion notification and fetch the ",
)
#: The reuse fragment lives in a Python string, so its rule is asserted
#: through literal-local fragments (the active fixture carries the joined
#: sentences, checked with TASK_RULE).
RENDERER_RULE_FRAGMENTS = (
    "end your turn and collect its result with `get_agent_result` when the ",
    "completion notification arrives, even if your next substantive action ",
    "depends on it. Use `wait_for_agent` only if the result is a hard ",
    "dependency of your next action, no independent work can proceed, and ",
    "that action must occur in this turn rather than after notification. If ",
    "any of those conditions is uncertain, end your turn. Do not invent ",
    "independent work or dispatch unnecessary tasks to fill the wait; if ",
    "`still_running` timeout is a normal outcome: end your turn or continue ",
    "do not re-wait in the same turn",
)

INVENTORY: tuple[Site, ...] = (
    Site(
        "task.md:9,:11,:31",
        "chartreux/core/tools/builtins/prompts/task.md",
        TASK_MD_OLD,
        TASK_MD_REQUIRED + TASK_RULE,
    ),
    Site(
        "fixture standalone/task.md:9,:11,:68",
        "tests/fixtures/dispatch/standalone/task.md",
        TASK_MD_OLD,
        TASK_MD_REQUIRED + TASK_RULE,
    ),
    Site(
        "wait_for_agent.md:1",
        "chartreux/core/tools/builtins/prompts/wait_for_agent.md",
        ("for a specific invocation, and optionally a positive `timeout`",),
        (NOTIFICATION_FIRST, '"digest"', '"full"') + WFA_RULE,
    ),
    Site(
        "cancel_agent.md:3",
        "chartreux/core/tools/builtins/prompts/cancel_agent.md",
        (
            "Use `wait_for_agent` or `get_agent_result` to retrieve the "
            "terminal result after cleanup",
        ),
        (NOTIFICATION_FIRST, MODE_DIGEST, MODE_FULL) + CANCEL_RULE,
    ),
    Site(
        "check_agents.md:1",
        "chartreux/core/tools/builtins/prompts/check_agents.md",
        (
            "Collect one with `get_agent_result(agent_id, run_id=...)` or "
            "`wait_for_agent`",
        ),
        (NOTIFICATION_FIRST, MODE_DIGEST, MODE_FULL) + CHECK_RULE,
    ),
    Site(
        "get_agent_result.py delivery-path pending message",
        "chartreux/core/tools/builtins/get_agent_result.py",
        (),
        ("a completion notification arrives", 'mode="digest"', 'mode="full"')
        + GAR_RULE,
    ),
    Site(
        "chartreux.py:728 (builtin skill prompt)",
        "chartreux/core/skills/builtins/chartreux.py",
        ("then wait for it with `wait_for_agent`", "wait if its result is needed"),
        (NOTIFICATION_FIRST, MODE_DIGEST, MODE_FULL) + BUILTIN_SKILL_RULE,
        skill_prompt=True,
    ),
    Site(
        "main-plan SKILL.md:27",
        "chartreux/skills/main-plan/SKILL.md",
        ("collect the advisor's result with `wait_for_agent`",),
        (NOTIFICATION_FIRST, MODE_DIGEST, MODE_FULL) + ADVISOR_SKILL_RULE,
    ),
    Site(
        "main-design SKILL.md:27",
        "chartreux/skills/main-design/SKILL.md",
        ("collect the advisor's result with `wait_for_agent`",),
        (NOTIFICATION_FIRST, MODE_DIGEST, MODE_FULL) + ADVISOR_SKILL_RULE,
    ),
    Site(
        "subagents.py:279-291,:711-718",
        "chartreux/core/subagents.py",
        (
            "re-invoke wait_for_agent in a later turn. The variants are mutually",
            "to keep waiting, or call check_agents to inspect progress.",
        ),
        (NOTIFICATION_FIRST, 'mode="digest"', 'mode="full"') + SUBAGENTS_RULE,
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
    """Both pending-path responses are reconditioned: the P2d delivery branch
    and the legacy (test-double) fallback teach notification-first collection
    and condition any wait on the three conditions.
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
        for clause in GAR_RULE:
            assert clause in branch

    def test_legacy_branch_is_reconditioned(self) -> None:
        # tests/tools/test_agent_management.py pins the legacy
        # (manager-without-_run_results) pending message verbatim; v0.7.9
        # rewrote that message and updated the pin in the same package, so the
        # old unconditional blocking advice survives nowhere.
        source = (
            ROOT / "chartreux/core/tools/builtins/get_agent_result.py"
        ).read_text()
        assert source.count("use wait_for_agent or try again later") == 0
        start = source.index("if results is None:")
        end = source.index("# P2d delivery states")
        branch = source[start:end]
        for clause in LEGACY_GAR_RULE:
            assert clause in branch


class TestPinnedCollisionSites:
    """Inventory sites v0.7.6 recorded as blocked by pins in files it did not
    own. v0.7.9 owns those pins and updates them in this package, so the sites
    assert the rewritten notification-first text instead of the old wording.
    """

    def test_launch_ack_carries_the_notification_first_rule(self) -> None:
        # tests/app_server/test_subagents_launch.py pins the full
        # acknowledgment by equality and test_subagent_management_surface.py
        # pins its prefix; both were updated with the rewrite in this package.
        source = (ROOT / "chartreux/app_server/_subagent_execution.py").read_text()
        for clause in ACK_RULE:
            assert clause in source
        assert "launched and running; use check_agents to monitor, " not in source
        assert "get_agent_result or wait_for_agent to retrieve the " not in source

    def test_renderer_reuse_block_carries_the_rule(self) -> None:
        # The fixture-mirror clause renders from core/dispatch/renderer.py's
        # _COMPOSITION_REUSE block; v0.7.9 replaced the running-agent sentence
        # with the same decision rule and regenerated the active fixtures.
        renderer = (ROOT / "chartreux/core/dispatch/renderer.py").read_text()
        fixture = (ROOT / "tests/fixtures/dispatch/standalone/task.md").read_text()
        for fragment in RENDERER_RULE_FRAGMENTS:
            assert fragment in renderer
        for clause in TASK_RULE:
            assert clause in fixture


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


class TestTaskPromptClauses:
    """The three separate task.md clauses, tested one by one — file-wide
    presence alone would let one occurrence carry the rule for all three.
    """

    @staticmethod
    def _clause(anchor: str) -> str:
        text = (ROOT / "chartreux/core/tools/builtins/prompts/task.md").read_text()
        start = text.index(anchor)
        end = text.find("\n\n", start)
        return text[start:] if end == -1 else text[start:end]

    def test_launch_clause_conditions_the_wait_and_foreground_exception(self) -> None:
        clause = self._clause(
            "Launch in the background even when the result is a dependency"
        )
        assert NOTIFICATION_FIRST in clause
        assert "even if your next substantive action depends on it" in clause
        assert TASK_RULE[1] in clause
        assert "If any of those conditions is uncertain, end your turn." in clause
        assert ANTI_INVENTION in clause
        assert "or block on the run with `wait_for_agent`" not in clause
        # The preserved foreground exception is true foreground semantics, not
        # a wait substitute for a background-eligible run.
        assert (
            "the caller's next action must consume the child's result in the "
            "current turn and the child cannot run detached" in clause
        )
        assert "never a wait-substitute for a background-eligible run" in clause
        assert "amendable mid-flight" in clause

    def test_retention_clause_has_no_second_blocking_alternative(self) -> None:
        clause = self._clause("Retention is opt-in.")
        assert "get_agent_result" in clause
        assert MODE_DIGEST in clause and MODE_FULL in clause
        assert "until result expiry" in clause
        assert "wait_for_agent" not in clause
        assert "retain: true" in clause

    def test_cancellation_clause_keeps_facts_and_conditions_the_wait(self) -> None:
        clause = self._clause("Use `cancel_agent(agent_id, run_id)`")
        assert "Acceptance means stop requested, not terminal cancellation." in clause
        assert "That notification-first fetch is the default collection path" in clause
        assert (
            "only if the result is a hard dependency of your next action, no "
            "independent work can proceed, and that action must occur in this "
            "turn rather than after notification; if any of those conditions is "
            "uncertain, end your turn" in clause
        )
        assert "never re-invoke the wait in the same turn" in clause
        assert "Use `release_agent(agent_id)`" in clause
        assert "Releasing an agent discards its retained results" in clause
        assert "Background agents cannot be launched from subagents." in clause
        assert MODE_DIGEST in clause and MODE_FULL in clause
