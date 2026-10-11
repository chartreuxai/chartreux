"""Documentation assertions for the tools-safety dangerous-command policy.

Keeps the security-normative claims of docs/guides/tools-safety.md honest:
the three policy classes, the grant choices, and the denylist breaking
change must stay documented as the runtime behaves.
"""

from __future__ import annotations

from pathlib import Path

DOC = (Path(__file__).parents[2] / "docs" / "guides" / "tools-safety.md").read_text()


def test_documents_three_policy_classes() -> None:
    assert "**Publication** (`git.push`" in DOC
    assert "**Consequential**" in DOC
    assert "git.push.force" in DOC
    assert "git.stash.clear" in DOC
    assert "**Hard guards**" in DOC
    assert "no operation ID and no grant path" in DOC


def test_documents_grant_choices() -> None:
    assert (
        "Deny (default), Allow once, Allow this session, or Always for this project"
        in DOC
    )
    assert (
        'Deny (default), Allow once, or Allow this session. **Never** "Always"' in DOC
    )
    assert (
        "a push combining consequential mechanisms requires all classified consequential operations"
        in DOC
    )


def test_documents_once_session_and_refusal_semantics() -> None:
    assert "exactly one invocation of the original call" in DOC
    assert "never shared across concurrent identical calls" in DOC
    assert "Refusal memory" in DOC
    assert "reformulated command re-prompts" in DOC
    assert "Recheck is the final gate" in DOC


def test_documents_breaking_change_and_absolute_denylist_veto() -> None:
    assert "### Breaking change" in DOC
    assert "`denylist = []` no longer implicitly authorizes" in DOC
    assert "Explicit `denylist` entries remain an absolute veto" in DOC


def test_documents_panel_and_no_public_grant_endpoint() -> None:
    assert "Settings > Shell Permissions" in DOC
    assert 'no public "grant an arbitrary operation" endpoint' in DOC
    assert "permission authority, not standing intent to publish" in DOC


def test_documents_bash_and_bash_start_parity() -> None:
    assert "`bash` and `bash_start` behave identically" in DOC
