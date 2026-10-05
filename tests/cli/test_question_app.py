from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container

from chartreux.app_server.models import (
    QuestionChoice,
    UserQuestion,
    UserQuestionRequest,
)
from chartreux.cli.textual_ui.widgets.question_app import QuestionApp


@pytest.fixture
def single_question_args():
    return UserQuestionRequest(
        questions=[
            UserQuestion(
                question="Which database?",
                header="DB",
                options=[
                    QuestionChoice(label="PostgreSQL", description="Relational DB"),
                    QuestionChoice(label="MongoDB", description="Document DB"),
                ],
            )
        ]
    )


@pytest.fixture
def multi_question_args():
    return UserQuestionRequest(
        questions=[
            UserQuestion(
                question="Which database?",
                header="DB",
                options=[
                    QuestionChoice(label="PostgreSQL"),
                    QuestionChoice(label="MongoDB"),
                ],
            ),
            UserQuestion(
                question="Which framework?",
                header="Framework",
                options=[
                    QuestionChoice(label="FastAPI"),
                    QuestionChoice(label="Django"),
                ],
            ),
        ]
    )


@pytest.fixture
def multi_select_args():
    return UserQuestionRequest(
        questions=[
            UserQuestion(
                question="Which features?",
                header="Features",
                options=[
                    QuestionChoice(label="Auth"),
                    QuestionChoice(label="Caching"),
                    QuestionChoice(label="Logging"),
                ],
                multi_select=True,
            )
        ]
    )


class TestQuestionAppState:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "removed_selector", ["#question-app", "#question-content", "#question-body"]
    )
    async def test_deferred_body_height_after_removal(
        self, single_question_args, removed_selector
    ):
        class Host(App[None]):
            def compose(self) -> ComposeResult:
                yield QuestionApp(single_question_args)

        app = Host()
        async with app.run_test() as pilot:
            picker = app.query_one(QuestionApp)
            await pilot.pause()
            max_height = picker._body_max_height
            # Hold the scheduled callback until removal has completed.
            with patch.object(picker, "call_after_refresh") as schedule:
                picker._schedule_body_max_height()
            schedule.assert_called_once()
            callback = schedule.call_args.args[0]
            await app.query_one(removed_selector).remove()
            app.call_after_refresh(callback)
            await pilot.pause()
            assert picker._body_max_height == max_height

    @pytest.mark.asyncio
    async def test_ascii_chrome_picker_prefixes(self, multi_select_args):
        class Host(App[None]):
            config: SimpleNamespace

            def compose(self) -> ComposeResult:
                yield QuestionApp(multi_select_args)

        app = Host()
        app.config = SimpleNamespace(ascii_chrome=True)
        async with app.run_test() as pilot:
            picker = app.query_one(QuestionApp)
            assert picker._format_option_prefix(0, True, True, True) == "  [x] "
            assert picker._format_option_prefix(0, True, False, False) == "> 1. "
            assert picker.submit_widget is not None
            assert str(picker.submit_widget.render()).endswith("->")
            await pilot.pause()

    def test_init_state(self, single_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)

        assert app.current_question_idx == 0
        assert app.selected_option == 0
        assert len(app.answers) == 0
        assert len(app.other_texts) == 0

    def test_more_than_four_options_accepted_and_rendered(self):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        args = UserQuestionRequest(
            questions=[
                UserQuestion(
                    question="Pick an analysis type",
                    header="Analysis",
                    options=[
                        QuestionChoice(label="Financial"),
                        QuestionChoice(label="Operational"),
                        QuestionChoice(label="Strategic"),
                        QuestionChoice(label="Competitive"),
                        QuestionChoice(label="Multi-source"),
                    ],
                )
            ]
        )

        app = QuestionApp(args)

        assert app.max_options == 5
        # 5 options + Other = 6 (no Submit for single-select)
        assert app._total_options == 6

    def test_total_options_single_select(self, single_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)

        # 2 options + Other = 3 (no Submit for single-select)
        assert app._total_options == 3

    def test_space_toggles_multi_select_without_advancing(self, multi_select_args):
        app = QuestionApp(multi_select_args)
        app.action_toggle_option()
        assert app.multi_selections[0] == {0}
        assert app.answers == {}
        app.action_toggle_option()
        assert app.multi_selections[0] == set()

    def test_format_multi_select_option_prefix_matches_checklist(
        self, multi_select_args
    ):
        app = QuestionApp(multi_select_args)

        assert app._format_option_prefix(0, True, True, True) == "  [■] "
        assert app._format_option_prefix(1, False, True, False) == "  [ ] "

    def test_total_options_multi_select_includes_submit(self, multi_select_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_select_args)

        # 3 options + Other + Submit = 5
        assert app._total_options == 5
        assert app._other_option_idx == 3
        assert app._submit_option_idx == 4

    def test_is_other_selected(self, single_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)

        assert app._is_other_selected is False
        app.selected_option = 2  # Other option
        assert app._is_other_selected is True

    def test_focusing_other_input_selects_other_option(self, single_question_args):
        from textual import events
        from textual.widgets import Input

        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        app.other_input = Input()
        app.selected_option = 0  # A regular option is selected

        app.on_descendant_focus(events.DescendantFocus(app.other_input))

        assert app.selected_option == app._other_option_idx
        assert app._is_other_selected is True

    def test_focusing_non_other_widget_leaves_selection(self, single_question_args):
        from textual import events
        from textual.widgets import Input

        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        app.other_input = Input()
        app.selected_option = 1

        app.on_descendant_focus(events.DescendantFocus(Input()))

        assert app.selected_option == 1

    def test_is_submit_selected(self, multi_select_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_select_args)

        assert app._is_submit_selected is False
        app.selected_option = 4  # Submit option
        assert app._is_submit_selected is True

    def test_is_submit_selected_false_for_single_select(self, single_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)

        # Even if selected_option is 3, is_submit_selected is False for single-select
        app.selected_option = 3
        assert app._is_submit_selected is False

    def test_store_other_text_per_question(self, multi_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_question_args)

        # Store text for question 0
        app.other_texts[0] = "Custom DB"

        # Switch to question 1
        app.current_question_idx = 1
        app.other_texts[1] = "Custom Framework"

        # Verify both stored separately
        assert app._get_other_text(0) == "Custom DB"
        assert app._get_other_text(1) == "Custom Framework"

    def test_save_regular_option_answer(self, single_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        app.selected_option = 0  # PostgreSQL

        app._save_current_answer()

        assert 0 in app.answers
        answer_text, is_other = app.answers[0]
        assert answer_text == "PostgreSQL"
        assert is_other is False

    def test_save_other_option_answer(self, single_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        app.selected_option = 2  # Other
        app.other_texts[0] = "SQLite"

        app._save_current_answer()

        assert 0 in app.answers
        answer_text, is_other = app.answers[0]
        assert answer_text == "SQLite"
        assert is_other is True

    def test_save_other_option_empty_does_not_save(self, single_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        app.selected_option = 2  # Other
        app.other_texts[0] = ""  # Empty

        app._save_current_answer()

        assert 0 not in app.answers

    def test_all_answered_false_initially(self, multi_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_question_args)

        assert app._all_answered() is False

    def test_all_answered_true_when_complete(self, multi_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_question_args)
        app.answers[0] = ("PostgreSQL", False)
        app.answers[1] = ("FastAPI", False)

        assert app._all_answered() is True

    def test_multi_select_toggle(self, multi_select_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_select_args)

        # Initially no selections
        assert len(app.multi_selections.get(0, set())) == 0

        # Add selection
        app.multi_selections.setdefault(0, set()).add(0)
        assert 0 in app.multi_selections[0]

        # Add another
        app.multi_selections[0].add(2)
        assert 2 in app.multi_selections[0]

        # Remove first
        app.multi_selections[0].discard(0)
        assert 0 not in app.multi_selections[0]
        assert 2 in app.multi_selections[0]

    def test_multi_select_save_answer(self, multi_select_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_select_args)
        app.multi_selections[0] = {0, 2}  # Auth and Logging

        app._save_current_answer()

        assert 0 in app.answers
        answer_text, is_other = app.answers[0]
        assert "Auth" in answer_text
        assert "Logging" in answer_text
        assert is_other is False

    def test_multi_select_with_other(self, multi_select_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_select_args)
        app.multi_selections[0] = {0, 3}  # Auth and Other
        app.other_texts[0] = "Custom Feature"

        app._save_current_answer()

        assert 0 in app.answers
        answer_text, is_other = app.answers[0]
        assert "Auth" in answer_text
        assert "Custom Feature" in answer_text
        assert is_other is True


class TestQuestionAppActions:
    def test_action_move_down(self, single_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        assert app.selected_option == 0

        app.action_move_down()
        assert app.selected_option == 1

        app.action_move_down()
        assert app.selected_option == 2  # Other

        app.action_move_down()
        assert app.selected_option == 0  # Wraps around

    def test_action_move_up(self, single_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        assert app.selected_option == 0

        app.action_move_up()
        assert app.selected_option == 2  # Wraps to Other

        app.action_move_up()
        assert app.selected_option == 1

    def test_switch_question_preserves_other_text(self, multi_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_question_args)
        app.other_texts[0] = "Text for Q1"

        app._switch_question(1)

        assert app.current_question_idx == 1
        assert app._get_other_text(0) == "Text for Q1"
        assert app._get_other_text(1) == ""

    def test_switch_question_restores_cursor_to_answered_option(
        self, multi_question_args
    ):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_question_args)
        # Answer question 0 with MongoDB (index 1)
        app.answers[0] = ("MongoDB", False)

        # Switch away then back
        app._switch_question(1)
        app._switch_question(0)

        assert app.selected_option == 1

    def test_switch_question_restores_cursor_to_other(self, multi_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_question_args)
        # Answer question 0 with Other
        app.answers[0] = ("Custom DB", True)

        app._switch_question(1)
        app._switch_question(0)

        assert app.selected_option == len(app.questions[0].options)
        assert app._is_other_selected

    def test_switch_question_defaults_to_zero_if_unanswered(self, multi_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_question_args)

        app._switch_question(1)
        assert app.selected_option == 0

    def test_switch_question_restores_cursor_multi_select(self):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        args = UserQuestionRequest(
            questions=[
                UserQuestion(
                    question="Q1?",
                    options=[QuestionChoice(label="A"), QuestionChoice(label="B")],
                ),
                UserQuestion(
                    question="Q2?",
                    options=[
                        QuestionChoice(label="X"),
                        QuestionChoice(label="Y"),
                        QuestionChoice(label="Z"),
                    ],
                    multi_select=True,
                ),
            ]
        )
        app = QuestionApp(args)
        # Select Y (1) and Z (2) for question 1
        app.multi_selections[1] = {1, 2}

        app._switch_question(1)

        assert app.selected_option == 1  # min of {1, 2}


class TestMultiSelectOtherBehavior:
    def test_multi_select_other_does_not_advance_on_save(self, multi_select_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_select_args)
        app.selected_option = 3  # Other option (3 options + Other)
        app.other_texts[0] = "Custom feature"

        # Save should not advance for multi-select
        app._save_current_answer()

        # Should stay on same question
        assert app.current_question_idx == 0

    def test_multi_select_other_toggle_adds_to_selections(self, multi_select_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_select_args)
        other_idx = len(app._current_question.options)  # 3

        # Initially no selections
        assert len(app.multi_selections.get(0, set())) == 0

        # Add Other to selections
        app.multi_selections.setdefault(0, set()).add(other_idx)
        app.other_texts[0] = "Custom"

        # Can still add regular options
        app.multi_selections[0].add(0)  # Auth
        app.multi_selections[0].add(1)  # Caching

        assert other_idx in app.multi_selections[0]
        assert 0 in app.multi_selections[0]
        assert 1 in app.multi_selections[0]

    def test_multi_select_save_with_other_and_regular_options(self, multi_select_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_select_args)
        other_idx = len(app._current_question.options)

        # Select Auth (0), Logging (2), and Other (3)
        app.multi_selections[0] = {0, 2, other_idx}
        app.other_texts[0] = "Custom Feature"

        app._save_current_answer()

        assert 0 in app.answers
        answer_text, is_other = app.answers[0]
        assert "Auth" in answer_text
        assert "Logging" in answer_text
        assert "Custom Feature" in answer_text
        assert is_other is True

    def test_multi_select_other_without_text_not_in_answer(self, multi_select_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_select_args)
        other_idx = len(app._current_question.options)

        # Select Auth (0) and Other (3) but no text for Other
        app.multi_selections[0] = {0, other_idx}
        app.other_texts[0] = ""

        app._save_current_answer()

        assert 0 in app.answers
        answer_text, is_other = app.answers[0]
        assert "Auth" in answer_text
        # Empty Other should not appear
        assert is_other is False  # No valid Other text

    def test_multi_select_can_toggle_after_selecting_other(self, multi_select_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_select_args)
        other_idx = len(app._current_question.options)

        # Select Other first
        app.multi_selections[0] = {other_idx}
        app.other_texts[0] = "Custom"

        # Now navigate to and toggle Auth
        app.selected_option = 0
        selections = app.multi_selections.setdefault(0, set())
        selections.add(0)  # Toggle Auth on

        assert 0 in app.multi_selections[0]
        assert other_idx in app.multi_selections[0]

        # Toggle Auth off
        selections.discard(0)
        assert 0 not in app.multi_selections[0]
        assert other_idx in app.multi_selections[0]

    def test_multi_select_empty_selections_does_not_save(self, multi_select_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_select_args)

        # No selections
        app._save_current_answer()

        assert 0 not in app.answers

    def test_multi_select_submit_without_selection_is_noop(self, multi_select_args):
        from unittest.mock import MagicMock

        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_select_args)
        app.selected_option = app._submit_option_idx
        app._submit = MagicMock()

        # Submitting with no option selected must not raise StopIteration/RuntimeError
        app._handle_multi_select_action()

        assert 0 not in app.answers
        app._submit.assert_not_called()


class TestSingleSelectOtherBehavior:
    def test_single_select_other_with_text_saves(self, single_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        app.selected_option = 2  # Other
        app.other_texts[0] = "Custom DB"

        app._save_current_answer()

        assert 0 in app.answers
        answer_text, is_other = app.answers[0]
        assert answer_text == "Custom DB"
        assert is_other is True

    def test_single_select_other_without_text_does_not_save(self, single_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        app.selected_option = 2  # Other
        app.other_texts[0] = ""

        app._save_current_answer()

        assert 0 not in app.answers

    def test_single_select_regular_option_saves_immediately(self, single_question_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        app.selected_option = 1  # MongoDB

        app._save_current_answer()

        assert 0 in app.answers
        answer_text, is_other = app.answers[0]
        assert answer_text == "MongoDB"
        assert is_other is False


class TestMultiSelectFreeChoice:
    @pytest.mark.asyncio
    async def test_enter_toggles_free_choice(self, multi_select_args):
        app = _HostApp(multi_select_args)
        async with app.run_test() as pilot:
            qapp = app.query_one(QuestionApp)
            other_idx = qapp._other_option_idx

            # Navigate with the keyboard so the free choice starts unselected.
            await pilot.press("down", "down", "down")
            assert qapp._is_other_selected
            assert other_idx not in qapp.multi_selections.get(0, set())

            await pilot.press("enter")
            assert other_idx in qapp.multi_selections[0]

            await pilot.press("enter")
            assert other_idx not in qapp.multi_selections[0]

    @pytest.mark.asyncio
    async def test_typing_selects_free_choice(self, multi_select_args):
        app = _HostApp(multi_select_args)
        async with app.run_test() as pilot:
            qapp = app.query_one(QuestionApp)
            other_idx = qapp._other_option_idx

            # Navigate with the keyboard so the free choice starts unselected.
            await pilot.press("down", "down", "down")
            assert other_idx not in qapp.multi_selections.get(0, set())

            await pilot.press(*"x")

            assert other_idx in qapp.multi_selections[0]

    @pytest.mark.asyncio
    async def test_typing_stores_text_and_keeps_selection(self, multi_select_args):
        app = _HostApp(multi_select_args)
        async with app.run_test() as pilot:
            qapp = app.query_one(QuestionApp)
            other_idx = qapp._other_option_idx

            assert qapp.other_static is not None
            await pilot.click(qapp.other_static)
            await pilot.press(*"custom")

            assert qapp.other_texts.get(0) == "custom"
            assert other_idx in qapp.multi_selections[0]

    @pytest.mark.asyncio
    async def test_clearing_free_choice_deselects_it(self, multi_select_args):
        app = _HostApp(multi_select_args)
        async with app.run_test() as pilot:
            qapp = app.query_one(QuestionApp)
            other_idx = qapp._other_option_idx

            assert qapp.other_static is not None
            await pilot.click(qapp.other_static)
            await pilot.press(*"hi")
            assert other_idx in qapp.multi_selections[0]

            await pilot.press("backspace", "backspace")

            assert qapp.other_texts.get(0) == ""
            assert other_idx not in qapp.multi_selections.get(0, set())


class TestNumberKeyShortcuts:
    def test_number_key_selects_predefined_option(self, single_question_args):
        from unittest.mock import patch

        from textual import events

        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        app.selected_option = 0

        # Press "2" -> should select MongoDB (index 1) and save
        event = events.Key("2", "2")
        with patch.object(app, "_advance_or_submit"):
            app._handle_number_key(event)

        assert app.selected_option == 1
        assert 0 in app.answers
        answer_text, is_other = app.answers[0]
        assert answer_text == "MongoDB"
        assert is_other is False

    def test_number_key_first_option(self, single_question_args):
        from unittest.mock import patch

        from textual import events

        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)

        event = events.Key("1", "1")
        with patch.object(app, "_advance_or_submit"):
            app._handle_number_key(event)

        assert app.selected_option == 0
        assert 0 in app.answers
        answer_text, _ = app.answers[0]
        assert answer_text == "PostgreSQL"

    def test_number_key_other_empty_focuses(self, single_question_args):
        from textual import events

        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        # Other is option index 2 -> key "3"
        event = events.Key("3", "3")
        app._handle_number_key(event)

        # Should navigate to Other but not validate (no text)
        assert app.selected_option == 2
        assert app._is_other_selected is True
        assert 0 not in app.answers

    def test_number_key_other_with_text_focuses_without_validating(
        self, single_question_args
    ):
        from unittest.mock import MagicMock

        from textual import events

        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        app.other_texts[0] = "SQLite"
        app.other_input = MagicMock()
        app.other_input.has_focus = False

        event = events.Key("3", "3")
        app._handle_number_key(event)

        # Should navigate to Other but not validate — just focus the input
        assert app.selected_option == 2
        assert app._is_other_selected is True
        assert 0 not in app.answers

    def test_number_key_out_of_range_ignored(self, single_question_args):
        from textual import events

        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        # 3 options total (2 + Other), so "4" is out of range
        event = events.Key("4", "4")
        app._handle_number_key(event)

        assert app.selected_option == 0
        assert 0 not in app.answers

    def test_number_key_zero_ignored(self, single_question_args):
        from textual import events

        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        event = events.Key("0", "0")
        app._handle_number_key(event)

        assert app.selected_option == 0
        assert 0 not in app.answers

    def test_number_key_ignored_when_input_focused(self, single_question_args):
        from unittest.mock import MagicMock

        from textual import events

        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        app.other_input = MagicMock()
        app.other_input.has_focus = True

        event = events.Key("1", "1")
        app._handle_number_key(event)

        assert app.selected_option == 0
        assert 0 not in app.answers

    def test_number_key_multi_select_toggles(self, multi_select_args):
        from textual import events

        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_select_args)

        # Press "1" in multi-select -> should toggle Auth
        event = events.Key("1", "1")
        app._handle_number_key(event)

        assert 0 in app.multi_selections.get(0, set())

    def test_number_key_multi_select_submit_ignored(self, multi_select_args):
        from textual import events

        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_select_args)
        # Submit is at index 4 -> key "5"
        event = events.Key("5", "5")
        app._handle_number_key(event)

        # Should not navigate to submit
        assert app.selected_option == 0


class TestMultiSelectSubmit:
    def test_navigate_to_submit(self, multi_select_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_select_args)

        # Navigate down through all options to Submit
        for _ in range(4):  # 0->1->2->3(Other)->4(Submit)
            app.action_move_down()

        assert app.selected_option == 4
        assert app._is_submit_selected is True

    def test_submit_wraps_around(self, multi_select_args):
        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(multi_select_args)
        app.selected_option = 4  # Submit

        app.action_move_down()

        assert app.selected_option == 0


class TestVimKeybindings:
    def test_j_moves_down(self, single_question_args):
        from textual import events

        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        assert app.selected_option == 0

        app.on_key(events.Key("j", "j"))

        assert app.selected_option == 1

    def test_k_moves_up(self, single_question_args):
        from textual import events

        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        assert app.selected_option == 0

        app.on_key(events.Key("k", "k"))

        # Wraps to last option (Other = index 2)
        assert app.selected_option == 2

    def test_j_k_ignored_when_other_input_focused(self, single_question_args):
        from unittest.mock import MagicMock

        from textual import events

        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        app = QuestionApp(single_question_args)
        app.other_input = MagicMock()
        app.other_input.has_focus = True

        app.on_key(events.Key("j", "j"))
        assert app.selected_option == 0

    def test_j_moves_down_for_single_question(self):
        # Vim navigation must fire before the len <= 1 early return in on_key.
        from textual import events

        from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

        args = UserQuestionRequest(
            questions=[
                UserQuestion(
                    question="Only question?",
                    header="Q",
                    options=[QuestionChoice(label="A"), QuestionChoice(label="B")],
                )
            ]
        )
        app = QuestionApp(args)
        assert app.selected_option == 0

        app.on_key(events.Key("j", "j"))

        assert app.selected_option == 1


class _HostApp(App[None]):
    CSS_PATH = "../../chartreux/cli/textual_ui/app.tcss"

    def __init__(self, args: UserQuestionRequest) -> None:
        super().__init__()
        self._args = args

    def compose(self) -> ComposeResult:
        with Container(id="bottom-app-container"):
            yield QuestionApp(args=self._args)


class TestQuestionAppClicks:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("gesture", ["click", "enter"])
    @pytest.mark.parametrize("size", [(80, 24), (120, 36)])
    async def test_acceptance_guards_advance_and_submit(
        self, multi_select_args, gesture, size
    ):
        args = multi_select_args.model_copy(deep=True)
        args.questions.append(args.questions[0].model_copy(deep=True))
        app = _HostApp(args)
        async with app.run_test(size=size) as pilot:
            qapp = app.query_one(QuestionApp)

            async def accept():
                assert qapp.submit_widget is not None
                if gesture == "click":
                    await pilot.click(qapp.submit_widget)
                else:
                    qapp.selected_option = qapp._submit_option_idx
                    await pilot.press("enter")

            with patch.object(
                qapp, "post_message", wraps=qapp.post_message
            ) as delivery:
                with patch.object(qapp, "is_within_grace_period", return_value=True):
                    qapp.multi_selections[0] = {0}
                    await accept()
                    assert qapp.answers == {}
                    assert qapp.current_question_idx == 0
                qapp._mount_time = 0.0
                qapp.multi_selections[0] = set()
                await accept()
                assert qapp.current_question_idx == 0
                assert qapp.answers == {}
                qapp.multi_selections[0] = {0}
                await accept()
                assert qapp.current_question_idx == 1
                assert qapp.answers[0] == ("Auth", False)
                assert not any(
                    isinstance(call.args[0], QuestionApp.Answered)
                    for call in delivery.call_args_list
                )
                qapp.multi_selections[1] = {1}
                await accept()
                await accept()
                answered = [
                    call.args[0]
                    for call in delivery.call_args_list
                    if isinstance(call.args[0], QuestionApp.Answered)
                ]
                assert len(answered) == 1
                assert [a.answer for a in answered[0].answers] == ["Auth", "Caching"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("gesture", ["click", "enter"])
    @pytest.mark.parametrize("invalid", ["empty", "blank_other"])
    async def test_revisited_invalid_acceptance_is_rejected(
        self, multi_select_args, gesture, invalid
    ):
        args = multi_select_args.model_copy(deep=True)
        args.questions.append(args.questions[0].model_copy(deep=True))
        app = _HostApp(args)
        async with app.run_test() as pilot:
            qapp = app.query_one(QuestionApp)
            qapp._mount_time = 0.0
            qapp.multi_selections[0] = {0}
            qapp.selected_option = qapp._submit_option_idx
            qapp.action_select()
            assert qapp.current_question_idx == 1
            qapp.action_prev_question()
            qapp.multi_selections[0] = (
                set() if invalid == "empty" else {qapp._other_option_idx}
            )
            qapp.other_texts[0] = "   "
            with patch.object(
                qapp, "post_message", wraps=qapp.post_message
            ) as delivery:
                if gesture == "click":
                    assert qapp.submit_widget is not None
                    await pilot.click(qapp.submit_widget)
                else:
                    qapp.selected_option = qapp._submit_option_idx
                    await pilot.press("enter")
                assert qapp.current_question_idx == 0
                assert 0 not in qapp.answers
                assert not any(
                    isinstance(call.args[0], QuestionApp.Answered)
                    for call in delivery.call_args_list
                )

    def test_definite_rejection_allows_resubmission(self, multi_select_args):
        qapp = QuestionApp(multi_select_args)
        qapp.multi_selections[0] = {0}
        qapp.selected_option = qapp._submit_option_idx
        with patch.object(qapp, "post_message", wraps=qapp.post_message) as delivery:
            qapp.action_select()
            qapp.set_submission_status("Failed: Answer rejected")
            qapp.action_select()
            assert delivery.call_count == 2

    @pytest.mark.asyncio
    async def test_clicking_option_selects_it(self, single_question_args):
        app = _HostApp(single_question_args)
        async with app.run_test() as pilot:
            qapp = app.query_one(QuestionApp)
            qapp._mount_time = 0.0
            assert qapp.selected_option == 0

            await pilot.click(qapp.option_widgets[1])

            assert qapp.selected_option == 1
            assert (
                qapp.answers[0][0] == single_question_args.questions[0].options[1].label
            )

    @pytest.mark.asyncio
    async def test_clicking_other_row_selects_other(self, single_question_args):
        app = _HostApp(single_question_args)
        async with app.run_test() as pilot:
            qapp = app.query_one(QuestionApp)
            assert not qapp._is_other_selected

            assert qapp.other_static is not None
            await pilot.click(qapp.other_static)

            assert qapp.selected_option == qapp._other_option_idx
            assert qapp._is_other_selected

    @pytest.mark.asyncio
    async def test_clicking_submit_selects_submit(self, multi_select_args):
        app = _HostApp(multi_select_args)
        async with app.run_test() as pilot:
            qapp = app.query_one(QuestionApp)
            assert qapp.submit_widget is not None

            await pilot.click(qapp.submit_widget)

            assert qapp.selected_option == qapp._submit_option_idx

    @pytest.mark.asyncio
    async def test_clicking_option_toggles_in_multi_select(self, multi_select_args):
        app = _HostApp(multi_select_args)
        async with app.run_test() as pilot:
            qapp = app.query_one(QuestionApp)

            await pilot.click(qapp.option_widgets[1])
            assert qapp.selected_option == 1
            assert 1 in qapp.multi_selections[0]

            await pilot.click(qapp.option_widgets[1])
            assert 1 not in qapp.multi_selections[0]

    @pytest.mark.asyncio
    async def test_clicking_free_choice_selects_it(self, multi_select_args):
        app = _HostApp(multi_select_args)
        async with app.run_test() as pilot:
            qapp = app.query_one(QuestionApp)

            assert qapp.other_static is not None
            await pilot.click(qapp.other_static)

            assert qapp.selected_option == qapp._other_option_idx
            assert qapp._other_option_idx in qapp.multi_selections[0]
