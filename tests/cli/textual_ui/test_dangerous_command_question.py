"""Dangerous-command question presentation in the question widget.

The P4 approval question arrives through the generic user-question widget.
These tests pin the frozen display contract: the literal command, project,
repository, consequence, and the publication-only "permission authority, not
a standing intent to publish" warning; the exact 4-option (publication) and
3-option (consequential) sets with Deny default-selected; Escape/cancel
failing closed to deny; markup-looking command text rendering literally; a
long command staying fully inspectable without hiding the choices; and
render-time secret redaction with a visible notice that never touches the
internal question identity.
"""

from __future__ import annotations

from pathlib import Path
import shlex

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll

from chartreux.cli.textual_ui.widgets.question_app import QuestionApp
from chartreux.core.agent_loop._dangerous_command_grants import (
    DANGEROUS_COMMAND_DENY_LABEL,
    DANGEROUS_COMMAND_ONCE_LABEL,
    DANGEROUS_COMMAND_PROJECT_LABEL,
    DANGEROUS_COMMAND_SESSION_LABEL,
    dangerous_command_request,
    parse_dangerous_command_result,
)
from chartreux.core.dangerous_commands import DangerousCommandProposal, RepositoryScope
from chartreux.core.tools import secret_redaction
from chartreux.questions import UserQuestionResult
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

PROJECT = Path("/home/example/projects/shop")
GIT_DIR = PROJECT / ".git"
WORK_TREE = PROJECT
SECRET = "ghp_SyntheticToken1234567890abcdef"


def _proposal(
    command: tuple[str, ...], operations: tuple[str, ...], consequence: str
) -> DangerousCommandProposal:
    return DangerousCommandProposal(
        command=command,
        project=PROJECT,
        scope=RepositoryScope(git_dir=GIT_DIR, work_tree=WORK_TREE),
        operations=operations,
        consequence=consequence,
    )


def _publication_proposal(
    command: tuple[str, ...] = ("git", "push", "origin", "main"),
) -> DangerousCommandProposal:
    return _proposal(
        command,
        ("git.push",),
        "Publish local commits to the remote (ordinary push: no forced, "
        "deleted, or mirrored refs).",
    )


class QuestionHost(App[None]):
    """Host that records the widget's answer/cancel messages."""

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.answered: list[QuestionApp.Answered] = []
        self.cancelled: list[QuestionApp.Cancelled] = []

    def compose(self) -> ComposeResult:
        yield QuestionApp(self.args)

    def on_question_app_answered(self, message: QuestionApp.Answered) -> None:
        self.answered.append(message)

    def on_question_app_cancelled(self, message: QuestionApp.Cancelled) -> None:
        self.cancelled.append(message)


class TestDangerousCommandRequest:
    """The P4-built question carries the frozen display contract."""

    def test_publication_request_shape(self) -> None:
        request = dangerous_command_request(_publication_proposal())
        assert len(request.questions) == 1
        question = request.questions[0]
        assert question.header == "Dangerous command"
        assert question.hide_other is True
        assert [option.label for option in question.options] == [
            DANGEROUS_COMMAND_DENY_LABEL,
            DANGEROUS_COMMAND_ONCE_LABEL,
            DANGEROUS_COMMAND_SESSION_LABEL,
            DANGEROUS_COMMAND_PROJECT_LABEL,
        ]
        text = question.question
        assert "Allow this dangerous git command?" in text
        assert shlex.join(("git", "push", "origin", "main")) in text
        assert f"Project: {PROJECT}" in text
        assert f"Repository: {WORK_TREE}" in text
        assert "Operations: git.push" in text
        assert "Consequence: Publish local commits" in text
        assert (
            "This grants permission authority for the operation, not a "
            "standing intent to publish." in text
        )
        assert request.footer_note is not None
        assert "user config" in request.footer_note

    def test_consequential_request_never_offers_persistence(self) -> None:
        proposal = _proposal(
            ("git", "push", "--force", "origin", "main"),
            ("git.push.force",),
            "Overwrite remote ref history (forced push).",
        )
        request = dangerous_command_request(proposal)
        question = request.questions[0]
        assert [option.label for option in question.options] == [
            DANGEROUS_COMMAND_DENY_LABEL,
            DANGEROUS_COMMAND_ONCE_LABEL,
            DANGEROUS_COMMAND_SESSION_LABEL,
        ]
        assert DANGEROUS_COMMAND_PROJECT_LABEL not in {
            option.label for option in question.options
        }
        assert "standing intent to publish" not in question.question
        assert request.footer_note is not None
        assert "never" in request.footer_note

    def test_mixed_operations_never_offer_persistence(self) -> None:
        proposal = _proposal(
            ("git", "push", "--force", "--delete", "origin", "main"),
            ("git.push.force", "git.push.delete"),
            "Overwrite and delete remote refs.",
        )
        request = dangerous_command_request(proposal)
        assert len(request.questions[0].options) == 3


class TestDangerousCommandPresentation:
    @pytest.mark.asyncio
    async def test_publication_question_renders_exactly_what_p4_sends(self) -> None:
        args = dangerous_command_request(_publication_proposal())
        app = QuestionHost(args)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            picker = app.query_one(QuestionApp)
            title = picker.query_one(".question-title", NoMarkupStatic)
            assert str(title.content) == args.questions[0].question
            assert "git push origin main" in str(title.content)
            assert "not a standing intent to publish" in str(title.content)
            labels = [
                str(widget.content)
                for widget in picker.option_widgets
                if widget.display
            ]
            cursor = chrome_glyph("cursor")
            assert labels == [
                f"{cursor} 1. {DANGEROUS_COMMAND_DENY_LABEL}"
                " - Skip this command; it does not run",
                f"  2. {DANGEROUS_COMMAND_ONCE_LABEL}"
                " - Run this exact command one time",
                f"  3. {DANGEROUS_COMMAND_SESSION_LABEL}"
                " - Allow these operations in this repository for this session",
                f"  4. {DANGEROUS_COMMAND_PROJECT_LABEL}"
                " - Allow for this session and save it to your user config "
                "for this project",
            ]
            # Deny is the default-selected option; the widget starts at zero.
            assert picker.selected_option == 0
            assert picker.option_widgets[0].has_class("question-option-selected")
            # hide_other: no free-text row is offered.
            assert picker.other_prefix is not None
            assert not picker.other_prefix.display

    @pytest.mark.asyncio
    async def test_deny_is_default_and_enter_answers_deny(self) -> None:
        args = dangerous_command_request(_publication_proposal())
        app = QuestionHost(args)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.answered) == 1
            answers = app.answered[0].answers
            assert len(answers) == 1
            assert answers[0].answer == DANGEROUS_COMMAND_DENY_LABEL
            assert answers[0].question == args.questions[0].question
            assert (
                parse_dangerous_command_result(
                    UserQuestionResult(answers=answers, cancelled=False), ("git.push",)
                )
                == "deny"
            )

    @pytest.mark.asyncio
    async def test_escape_cancels_and_parse_fails_closed_to_deny(self) -> None:
        args = dangerous_command_request(_publication_proposal())
        app = QuestionHost(args)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert len(app.cancelled) == 1
            assert not app.answered
        # The P4 parse of a cancelled result authorizes nothing.
        assert (
            parse_dangerous_command_result(
                UserQuestionResult(answers=[], cancelled=True), ("git.push",)
            )
            == "deny"
        )

    @pytest.mark.asyncio
    async def test_markup_looking_command_renders_literally(self) -> None:
        command = (
            "git",
            "push",
            "origin",
            "[red]main[/red]",
            "[bold]--force-with-lease[/bold]",
        )
        args = dangerous_command_request(_publication_proposal(command))
        app = QuestionHost(args)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            picker = app.query_one(QuestionApp)
            title = picker.query_one(".question-title", NoMarkupStatic)
            content = str(title.content)
            # No markup injection: the control-looking text is literal text.
            assert content == args.questions[0].question
            assert "[red]main[/red]" in content
            assert "[bold]--force-with-lease[/bold]" in content
            rendered = str(title.render())
            assert "[red]main[/red]" in rendered

    @pytest.mark.asyncio
    async def test_secret_redaction_notices_and_preserves_identity(self) -> None:
        command = (
            "git",
            "push",
            f"https://oauth2:{SECRET}@github.com/example/shop.git",
            "main",
        )
        args = dangerous_command_request(_publication_proposal(command))
        policy = secret_redaction.ScrubPolicy(
            redaction_credentials=(("TEST_TOKEN", SECRET),)
        )
        app = QuestionHost(args)
        with secret_redaction.bind_policy(policy):
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                picker = app.query_one(QuestionApp)
                title = picker.query_one(".question-title", NoMarkupStatic)
                notice = picker.query_one(".question-redaction-notice", NoMarkupStatic)
                displayed = str(title.content)
                # The rendered question never shows the secret value.
                assert SECRET not in displayed
                assert secret_redaction.REDACTED_PLACEHOLDER in displayed
                # The redaction is never silent: a visible notice says so.
                assert notice.display
                assert "redacted for display" in str(notice.content)
                assert "unchanged" in str(notice.content)
                # The internal identity stays exact.
                assert picker.questions[0].question == args.questions[0].question
                assert SECRET in picker.questions[0].question
                await pilot.press("enter")
                await pilot.pause()
                assert len(app.answered) == 1
                answer = app.answered[0].answers[0]
                assert answer.question == args.questions[0].question
                assert SECRET in answer.question
                assert answer.answer == DANGEROUS_COMMAND_DENY_LABEL

    @pytest.mark.asyncio
    async def test_long_command_fully_inspectable_without_hiding_choices(self) -> None:
        long_ref = "feature/" + "x" * 400
        command = ("git", "push", "origin", long_ref, "--verbose", "--progress")
        args = dangerous_command_request(_publication_proposal(command))
        app = QuestionHost(args)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            picker = app.query_one(QuestionApp)
            body = picker.query_one("#question-body", VerticalScroll)
            title = picker.query_one(".question-title", NoMarkupStatic)
            help_row = picker.query_one(".question-help", NoMarkupStatic)
            # The whole command is present and inspectable in the scroll body.
            assert str(title.content) == args.questions[0].question
            assert long_ref in str(title.content)
            assert body.max_scroll_y > 0
            # The choices stay reachable: navigation lands on every option and
            # the help row stays inside the panel.
            for expected in range(4):
                assert picker.selected_option == expected
                await pilot.press("down")
            assert picker.selected_option == 0  # wraps back to Deny
            await pilot.press("up")
            assert picker.selected_option == 3
            assert all(widget.display for widget in picker.option_widgets)
            assert help_row.display
            assert help_row.region.bottom <= picker.region.bottom
            # Answering the last option still works at 80x24.
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.answered) == 1
            assert app.answered[0].answers[0].answer == DANGEROUS_COMMAND_PROJECT_LABEL

    @pytest.mark.asyncio
    async def test_consequential_question_renders_three_options(self) -> None:
        proposal = _proposal(
            ("git", "stash", "drop"),
            ("git.stash.drop",),
            "Drop a stash entry; its changes leave the stash.",
        )
        args = dangerous_command_request(proposal)
        app = QuestionHost(args)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            picker = app.query_one(QuestionApp)
            labels = [
                str(widget.content)
                for widget in picker.option_widgets
                if widget.display
            ]
            assert len(labels) == 3
            deny_suffix = DANGEROUS_COMMAND_DENY_LABEL + (
                " - Skip this command; it does not run"
            )
            assert labels[0].endswith(deny_suffix)
            assert picker.selected_option == 0
            # Number keys reach the third option; a fourth does nothing.
            await pilot.press("3")
            await pilot.pause()
            assert len(app.answered) == 1
            assert app.answered[0].answers[0].answer == DANGEROUS_COMMAND_SESSION_LABEL
