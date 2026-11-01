"""O1: the completion-audit stamp section in the agent transcript viewer."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server.events import AgentsUpdate
from chartreux.app_server.models import (
    PublicClaimFidelity,
    PublicCompletionAudit,
    PublicDeclaredOutcome,
    PublicDisclosureCompleteness,
    PublicMandateSatisfaction,
    PublicRejectionReason,
    PublicRuntimeFailure,
)
from chartreux.app_server.protocol import (
    AgentSummaryModel,
    AgentTranscriptGetResponse,
    AgentTranscriptState,
)
from chartreux.cli.textual_ui.widgets.agent_transcript import (
    AgentTranscriptViewer,
    completion_audit_lines,
)
from chartreux.core.subagents import derive_agent_name
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from tests.conftest import build_test_chartreux_app

if TYPE_CHECKING:
    from chartreux.cli.textual_ui.app import ChartreuxApp

#: Every line the section may render. Anything else — a timestamp, a
#: duration, a run id — would be chronology leaking into the pane.
_AUDIT_LINE_PATTERN = re.compile(
    r"^(Completion Audit"
    r"|Audit version unsupported"
    r"|Claims (matched|unverified|contradicted)"
    r"|Disclosure (complete|incomplete)"
    r"|Mandates (satisfied|unaddressed|disabled)"
    r"|Worker-declared (completed|incomplete)"
    r"|Runtime failure: (denied|timeout|unavailable|unrunnable)"
    r"|Rejection: [a-z]+( [a-z]+)+)$"
)


def _stamp(**overrides: object) -> PublicCompletionAudit:
    values: dict[str, object] = {
        "schema_version": 1,
        "claim_fidelity": PublicClaimFidelity.CLAIM_MATCHED,
        "disclosure_completeness": PublicDisclosureCompleteness.DISCLOSURE_COMPLETE,
        "mandate_satisfaction": PublicMandateSatisfaction.MANDATE_SATISFIED,
        "declared_outcome": PublicDeclaredOutcome.OUTCOME_COMPLETED,
        "primary_runtime_failure": PublicRuntimeFailure.RUNTIME_HEALTHY,
    }
    values.update(overrides)
    return PublicCompletionAudit.model_validate(values)


def _agent(
    agent_id: str = "one",
    *,
    availability: str = "idle",
    audit: PublicCompletionAudit | None = None,
    **overrides: object,
) -> AgentSummaryModel:
    values: dict[str, object] = {
        "agent_id": agent_id,
        "identity": "coder",
        "availability": availability,
        "name": derive_agent_name(
            agent_id=agent_id,
            identity="coder",
            identity_display_label=None,
            base_model=None,
            effective_model=None,
        ),
        "completion_audit": audit,
    }
    values.update(overrides)
    return AgentSummaryModel.model_validate(values)


async def _publish_agent(app: ChartreuxApp, agent: AgentSummaryModel) -> None:
    app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
        return_value=AgentTranscriptGetResponse(
            state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
        )
    )
    await app._handle_turn_event(AgentsUpdate([agent]))


def _audit_widget(viewer: AgentTranscriptViewer) -> NoMarkupStatic:
    return viewer.query_one("#agent-transcript-audit", NoMarkupStatic)


def _rendered(viewer: AgentTranscriptViewer) -> str:
    return str(_audit_widget(viewer).render())


@pytest.mark.asyncio
async def test_audit_section_renders_in_the_metadata_before_transcript_entries() -> (
    None
):
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await _publish_agent(app, _agent(audit=_stamp()))
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        viewer = app.query_one(AgentTranscriptViewer)
        assert app.query_one(AgentTranscriptViewer) is viewer
        audit = _audit_widget(viewer)
        assert audit.display
        assert _rendered(viewer) == (
            "Completion Audit\n"
            "Claims matched\n"
            "Disclosure complete\n"
            "Mandates satisfied\n"
            "Worker-declared completed"
        )
        # PLACEMENT: the section is transcript METADATA, rendered before the
        # entries — a sibling of the entry container, never one of its rows.
        scroll = viewer.query_one("#agent-transcript-content-scroll")
        content = viewer.query_one("#agent-transcript-content")
        children = list(scroll.children)
        assert children.index(audit) < children.index(content)
        assert content not in list(audit.ancestors)
        # Wording: the stamp classifies claims; it never claims verification.
        rendered = _rendered(viewer)
        assert "Claims matched" in rendered
        assert "Task verified" not in rendered
        assert "verified" not in rendered.lower()


@pytest.mark.asyncio
async def test_audit_section_is_absent_without_a_stamp() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await _publish_agent(app, _agent())
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        viewer = app.query_one(AgentTranscriptViewer)
        audit = _audit_widget(viewer)
        assert not audit.display
        assert _rendered(viewer) == ""
        assert _audit_widget(viewer).display is False


@pytest.mark.asyncio
async def test_audit_unsupported_version_is_neutral() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await _publish_agent(app, _agent(audit=_stamp(schema_version=2)))
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        viewer = app.query_one(AgentTranscriptViewer)
        rendered = _rendered(viewer)
        assert rendered == "Completion Audit\nAudit version unsupported"
        for interpreted in ("Claims", "Disclosure", "Mandates", "Worker-declared"):
            assert interpreted not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "rejection", "expected", "unexpected"),
    [
        (
            PublicRuntimeFailure.RUNTIME_HEALTHY,
            None,
            [],
            ["Runtime failure", "Rejection:"],
        ),
        (
            PublicRuntimeFailure.RUNTIME_HEALTHY,
            PublicRejectionReason.CLAIM_CONTRADICTED,
            ["Rejection: claim contradicted"],
            ["Runtime failure"],
        ),
        (
            PublicRuntimeFailure.RUNTIME_TIMEOUT,
            None,
            ["Runtime failure: timeout"],
            ["Rejection:"],
        ),
        (
            PublicRuntimeFailure.RUNTIME_UNAVAILABLE,
            PublicRejectionReason.ENVELOPE_MALFORMED,
            ["Runtime failure: unavailable", "Rejection: envelope malformed"],
            [],
        ),
    ],
)
async def test_audit_failure_and_rejection_lines_only_when_present(
    failure: PublicRuntimeFailure,
    rejection: PublicRejectionReason | None,
    expected: list[str],
    unexpected: list[str],
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        stamp = _stamp(primary_runtime_failure=failure, rejection=rejection)
        await _publish_agent(app, _agent(audit=stamp))
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        viewer = app.query_one(AgentTranscriptViewer)
        rendered = _rendered(viewer)
        for line in expected:
            assert line in rendered
        for phrase in unexpected:
            assert phrase not in rendered
        # The three claim classes stay present in every combination.
        assert "Claims matched" in rendered
        assert "Disclosure complete" in rendered
        assert "Mandates satisfied" in rendered


@pytest.mark.asyncio
async def test_audit_section_clears_on_retask_and_never_shows_the_previous_run() -> (
    None
):
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await _publish_agent(
            app,
            _agent(audit=_stamp(claim_fidelity=PublicClaimFidelity.CLAIM_CONTRADICTED)),
        )
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        viewer = app.query_one(AgentTranscriptViewer)
        assert "Claims contradicted" in _rendered(viewer)
        assert _audit_widget(viewer).display

        # The SAME handle re-tasks: the new run's summary carries no stamp,
        # so the section clears immediately and the previous run's stamp is
        # never associated with the new run.
        await app._handle_turn_event(
            AgentsUpdate([
                _agent(availability="running", current_run_id="r2", audit=None)
            ])
        )
        await pilot.pause()
        assert app.query_one(AgentTranscriptViewer) is viewer
        assert not _audit_widget(viewer).display
        assert _rendered(viewer) == ""
        assert _audit_widget(viewer).display is False


@pytest.mark.asyncio
async def test_audit_section_clears_when_the_result_expires() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await _publish_agent(app, _agent(audit=_stamp()))
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        viewer = app.query_one(AgentTranscriptViewer)
        assert _audit_widget(viewer).display

        await app._handle_turn_event(
            AgentsUpdate([_agent(availability="idle", result_expired=True)])
        )
        await pilot.pause()
        assert not _audit_widget(viewer).display
        assert _rendered(viewer) == ""


@pytest.mark.asyncio
async def test_audit_section_leaves_with_the_released_agent() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await _publish_agent(app, _agent(audit=_stamp()))
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        assert app.query("#agent-transcript-audit")

        # A release removes the handle from the published inventory: the
        # viewer closes, and no audit section survives it.
        await app._handle_turn_event(AgentsUpdate([]))
        await pilot.pause()
        assert not app.query(AgentTranscriptViewer)
        assert not app.query("#agent-transcript-audit")


def test_audit_section_is_unbordered_and_reuses_the_existing_palette() -> None:
    css = AgentTranscriptViewer.DEFAULT_CSS
    block = css.split("#agent-transcript-audit")[1].split("}")[0]
    # Compact and unbordered: no border declarations at all.
    assert "border" not in block
    # No literal color values, and no palette entry the rest of the viewer
    # does not already use: the text is the signal, never a new channel.
    assert not re.search(r"#[0-9a-fA-F]{3,8}|rgba?\(|hsla?\(", block)
    elsewhere = css.replace(block, "")
    assert set(re.findall(r"\$[a-z-]+", block)) <= set(
        re.findall(r"\$[a-z-]+", elsewhere)
    )
    assert "$text-muted" in block


def test_audit_lines_carry_no_chronology() -> None:
    stamps = [
        _stamp(),
        _stamp(claim_fidelity=PublicClaimFidelity.CLAIM_UNVERIFIED),
        _stamp(claim_fidelity=PublicClaimFidelity.CLAIM_CONTRADICTED),
        _stamp(
            disclosure_completeness=PublicDisclosureCompleteness.DISCLOSURE_INCOMPLETE
        ),
        _stamp(mandate_satisfaction=PublicMandateSatisfaction.MANDATE_UNADDRESSED),
        _stamp(mandate_satisfaction=PublicMandateSatisfaction.MANDATE_DISABLED),
        _stamp(declared_outcome=PublicDeclaredOutcome.OUTCOME_INCOMPLETE),
        _stamp(primary_runtime_failure=PublicRuntimeFailure.RUNTIME_DENIED),
        _stamp(rejection=PublicRejectionReason.RECEIPT_UNVERIFIABLE),
        _stamp(schema_version=7),
    ]
    observed: set[str] = set()
    for stamp in stamps:
        lines = completion_audit_lines(stamp)
        assert lines[0] == "Completion Audit"
        for line in lines:
            assert _AUDIT_LINE_PATTERN.match(line), line
            observed.add(line)
    # Every claim class and every fault line is reachable through the closed
    # vocabulary: nothing is silently omitted or interpolated.
    assert {
        "Claims matched",
        "Claims unverified",
        "Claims contradicted",
        "Disclosure incomplete",
        "Mandates unaddressed",
        "Mandates disabled",
        "Worker-declared incomplete",
        "Runtime failure: denied",
        "Rejection: receipt unverifiable",
        "Audit version unsupported",
    } <= observed
    assert completion_audit_lines(None) == []


@pytest.mark.asyncio
async def test_audit_section_does_not_enter_the_parent_facing_summary() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await _publish_agent(
            app, _agent(audit=_stamp(), current_task_summary="do the thing")
        )
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        viewer = app.query_one(AgentTranscriptViewer)
        summary = app._agent_browser.agent_summaries[0]
        assert summary.current_task_summary == "do the thing"
        assert "Completion Audit" not in summary.current_task_summary
        assert "Claims matched" not in summary.current_task_summary
        # The metadata pane keeps the run description; the stamp lives in its
        # own section and never rewrites the summary text.
        metadata = str(viewer.query_one(".agent-transcript-metadata").render())
        assert "Current task: do the thing" in metadata
        assert "Claims matched" not in metadata
