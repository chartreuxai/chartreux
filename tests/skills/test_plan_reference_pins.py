"""P10 chunk 9 (PR-05/PR-21): the shipped skill documents the plan surface.

The `chartreux` skill prompt is model context: a subcommand it never names is
one the model will not recommend, and stale prose teaches a surface that no
longer exists. These POSITIVE PINS assert the shipped text names every part of
the plan-family surface this package landed — `/plan status`, the recovery
actions, the rebind flow, the consent-modal field set, and the logging-disabled
entry refusal — so a zero-diff prose change cannot pass the acceptance.

`tests/cli/test_commands.py` guards the token direction (every documented slash
token is registered); this file guards the prose direction (the documented
surface still matches the runtime). The shipped prompt wraps hard, so the pins
compare whitespace-normalized text.
"""

from __future__ import annotations

import re

from chartreux.core.skills.builtins.chartreux import SKILL

_WHITESPACE = re.compile(r"\s+")


def _shipped_text() -> str:
    return _WHITESPACE.sub(" ", SKILL.prompt)


def test_the_entry_line_names_every_plan_family_subcommand() -> None:
    text = _shipped_text()
    assert (
        "`/plan`, `/plan status`, `/plan accept`, `/plan cancel`, `/plan rebind`"
        in text
    )


def test_status_reports_the_episode_state_and_its_document() -> None:
    text = _shipped_text()
    assert "`/plan status`" in text
    assert "reports the episode identity and its observed state" in text
    assert "along with the plan document and the recovery actions" in text
    assert "a terminal or unidentifiable episode is still displayed" in text


def test_status_names_every_observed_episode_state() -> None:
    text = _shipped_text()
    for state in (
        "draft",
        "accepting",
        "accepted",
        "cancelled",
        "possibly-executed",
        "executed",
        "legacy-unverified",
        "unresolvable",
    ):
        assert state in text, state


def test_the_accept_recovery_and_adjudication_actions_are_documented() -> None:
    text = _shipped_text()
    assert "re-offers it to the owning session" in text
    assert "release the claim and restart" in text
    assert "verify the surviving bytes and re-consent" in text
    assert "accept is disabled on a legacy-unverified record" in text
    assert "`/plan mark-accepted`" in text
    assert "inspecting the recorded executor session" in text
    assert "a cancel that restores the pre-entry selections" in text


def test_the_rebind_flow_is_documented() -> None:
    text = _shipped_text()
    assert "`/plan rebind <path>`" in text
    assert "inside the authorized roots" in text
    assert "the session cwd or chartreux plan storage" in text
    assert "refuses an outside path" in text
    assert "requires a fresh consent" in text


def test_the_consent_modal_fields_are_documented() -> None:
    text = _shipped_text()
    for field in (
        "the episode",
        "its lifecycle state",
        "the plan document",
        "the theme",
        "the snapshotted executor selection",
    ):
        assert field in text, field
    assert "the document content, summary, size, SHA-256 digest" in text
    assert "seeded with exactly the confirmed bytes" in text
    assert "Cancel or Escape (the default)" in text


def test_the_logging_disabled_entry_refusal_is_documented() -> None:
    text = _shipped_text()
    assert "refuses to start, and a stored episode refuses to resume" in text
    assert "when session logging is disabled" in text
    assert "depends on the logging domain" in text
