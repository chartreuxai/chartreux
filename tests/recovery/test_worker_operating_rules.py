from __future__ import annotations

from pathlib import Path

from chartreux.core.agents import AgentManager
from chartreux.core.completion_contract import (
    CompletionArtifactKind,
    CompletionCheckTemplate,
)
from chartreux.core.skills.manager import SkillManager
from chartreux.core.system_prompt import get_universal_system_prompt
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

ROOT = Path(__file__).resolve().parents[2]

CHILD_RULES_FRAGMENT = ROOT / "chartreux/core/prompts/fragments/bounded_child.md"
CODER_IDENTITY = ROOT / "chartreux/core/prompts/identities/coder.md"


def _child_prompt() -> str:
    # The composed bounded-child prompt: the retired worker.md body's
    # successor is the bounded-child context fragment plus the tool-semantics
    # fragment, spliced by the nine-step composition.
    config = build_test_vibe_config(
        include_model_info=False,
        include_prompt_detail=False,
        include_project_context=False,
        include_commit_signature=False,
    )
    return str(
        get_universal_system_prompt(
            config,
            SkillManager(lambda: config),
            AgentManager(FakeConfigOrchestrator(config)),
            is_subagent=True,
        )
    )


def test_child_prompt_includes_the_operating_rules() -> None:
    rendered = _child_prompt()

    # The bounded-child context fragment carries the subagent contract and
    # the child operating rules.
    assert "You are a bounded child agent" in rendered
    assert "## Child operating rules" in rendered
    assert "1. Denials: record each denied command" in rendered
    assert "2. Strategy switch: after the second denial" in rendered
    # The tool-semantics fragment carries the shell grammar into every prompt.
    assert (
        "Shell grammar: the shell has no variable expansion, command substitution, or loops; "
        "commands must be literal. Environment assignments prefixing commands must use "
        "uppercase names. No leading `=` in command names. Run scripts through an explicit "
        "interpreter, not implicit `uv`."
    ) in rendered


def test_bash_read_description_exposes_schema_limits() -> None:
    description = (
        ROOT / "chartreux/core/tools/builtins/prompts/bash_read.md"
    ).read_text(encoding="utf-8")

    assert (
        "`max_bytes` must be 4096-64000 UTF-8 output bytes (minimum 4096)."
        in description
    )
    assert "`wait_seconds` must not exceed 30." in description


def test_child_denial_and_authority_rules_are_served_verbatim() -> None:
    # The denial/authority rules the retired worker.md body pinned survive in
    # the bounded-child fragment and are served unmodified by the composition.
    rendered = _child_prompt()
    fragment = CHILD_RULES_FRAGMENT.read_text(encoding="utf-8")

    for sentence in (
        "1. Denials: record each denied command and the runtime's stated reason",
        "2. Strategy switch: after the second denial citing the same runtime rule/pattern",
        "3. Destinations: resolve log, temporary-file, report, and build-output destinations",
        "4. Provider isolation: external-provider calls are calls to remote services",
        "5. Edited paths: maintain a cumulative record of every repository path",
        "6. Scope reconciliation: before returning, compare the edited-path record",
    ):
        assert sentence in fragment
        assert sentence in rendered


def test_child_evidence_rules_are_served_verbatim() -> None:
    rendered = _child_prompt()
    fragment = CHILD_RULES_FRAGMENT.read_text(encoding="utf-8")

    for sentence in (
        "7. Evidence-linked work loop: the default pattern for every assignment",
        "8. Receipts and tool-result trust: every tool result you issued is evidence",
        "9. No-edit proof obligation: a claim that nothing changed",
        "10. File-state metadata is not authority:",
        "11. Completion citations: when the assignment produced a commit",
        "12. Completion envelope emission: when a completion contract frozen at admission",
        # The post-B exemption pin (binding): evidence obligations never
        # create an unconditional commit mandate for exempt/read-only seats.
        "never create an unconditional commit mandate for exempt or read-only assignments",
        # Envelope emission keyed to the P4 render.
        "Completion contract (frozen at admission):",
        "envelope required: yes",
        "completion_envelope_v1",
    ):
        assert sentence in fragment
        assert sentence in rendered


def test_child_envelope_instruction_is_keyed_to_the_rendered_contract() -> None:
    from chartreux.app_server._subagent_records import (
        CompletionCapability,
        FrozenCompletionContract,
    )
    from chartreux.core.recovery_models import PackageTier

    contract = FrozenCompletionContract.freeze(
        identity="coder",
        identity_revision=None,
        seat=None,
        tier=PackageTier.SUBSTANTIVE,
        cwd="/workspace",
        roots=("/workspace",),
        completion_check=[CompletionCheckTemplate.PYTEST],
        required_artifacts=[CompletionArtifactKind.COMMIT],
        shell_less=False,
        audit_mode="stamp",
        capability=CompletionCapability.AVAILABLE,
    )
    section = contract.render_prompt_section()
    assert "Completion contract (frozen at admission):" in section
    assert "- envelope required: yes" in section
    assert "- schema version: 1" in section
    # The served instruction keys on the exact strings the P4 render emits.
    fragment = CHILD_RULES_FRAGMENT.read_text(encoding="utf-8")
    for key in (
        "Completion contract (frozen at admission):",
        "envelope required: yes",
        "completion_envelope_v1",
    ):
        assert key in fragment


def test_coder_identity_carries_the_evidence_linked_completion_rules() -> None:
    body = CODER_IDENTITY.read_text(encoding="utf-8")
    rendered = _child_prompt()

    for sentence in (
        "**Evidence-linked completion**",
        "emit the strict completion receipt",
        "at least the seven-character prefix",
    ):
        assert sentence in body
        assert sentence in rendered
    # The lean coder identity keeps its model-tier ban: the completion
    # vocabulary never names tiers.
    for anchor in (
        "mechanical-tier",
        "model-tier",
        "package tier",
        "substantive package",
    ):
        assert anchor not in body
