from __future__ import annotations

from pydantic import ValidationError
import pytest

from chartreux.core.events import ToolCallEvent, ToolResultEvent, UserInputOrigin
from chartreux.core.tools.base import BaseToolState, InvokeContext, ToolError
from chartreux.core.tools.builtins.ask_user_question import (
    AskUserQuestion,
    AskUserQuestionArgs,
    AskUserQuestionConfig,
    AskUserQuestionResult,
)
from chartreux.questions import (
    QuestionChoice as Choice,
    QuestionKind,
    UnansweredQuestion,
    UnansweredQuestions,
    UnattendedDefault,
    UnattendedDefaultAction,
    UserAnswer as Answer,
    UserQuestion as Question,
    UserQuestionResult,
)
from tests.stubs.fake_interaction_requests import FakeInteractionRequests


@pytest.fixture
def tool():
    config = AskUserQuestionConfig()
    return AskUserQuestion(config_getter=lambda: config, state=BaseToolState())


@pytest.fixture
def single_question_args():
    return AskUserQuestionArgs(
        questions=[
            Question(
                question="Which database?",
                header="DB",
                options=[
                    Choice(label="PostgreSQL", description="Relational DB"),
                    Choice(label="MongoDB", description="Document DB"),
                ],
            )
        ]
    )


@pytest.fixture
def multi_question_args():
    return AskUserQuestionArgs(
        questions=[
            Question(
                question="Which database?",
                header="DB",
                options=[Choice(label="PostgreSQL"), Choice(label="MongoDB")],
            ),
            Question(
                question="Which framework?",
                header="Framework",
                options=[Choice(label="FastAPI"), Choice(label="Django")],
            ),
        ]
    )


async def run_tool_with_requests(tool, args, handler):
    ctx = InvokeContext(
        interaction_requests=FakeInteractionRequests(user_input=handler),
        tool_call_id="123",
    )
    result = None
    async for item in tool.run(args, ctx):
        result = item
    return result


@pytest.mark.asyncio
async def test_raises_error_without_interaction_requests(tool, single_question_args):
    with pytest.raises(ToolError) as err:
        async for _ in tool.run(single_question_args, ctx=None):
            pass

    assert "interactive UI" in str(err.value)


@pytest.mark.asyncio
async def test_requests_user_input_and_returns_result(tool, single_question_args):
    expected_result = AskUserQuestionResult(
        answers=[
            Answer(question="Which database?", answer="PostgreSQL", is_other=False)
        ],
        cancelled=False,
    )

    async def respond(args):
        assert args == single_question_args
        return expected_result

    result = await run_tool_with_requests(tool, single_question_args, respond)

    assert result is not None
    assert result == expected_result
    assert result.answers[0].answer == "PostgreSQL"


@pytest.mark.asyncio
async def test_handles_cancelled_result(tool, single_question_args):
    expected_result = AskUserQuestionResult(answers=[], cancelled=True)

    async def respond(args):
        return expected_result

    result = await run_tool_with_requests(tool, single_question_args, respond)

    assert result is not None
    assert result.cancelled is True
    assert len(result.answers) == 0


@pytest.mark.asyncio
async def test_handles_other_response(tool, single_question_args):
    expected_result = AskUserQuestionResult(
        answers=[Answer(question="Which database?", answer="SQLite", is_other=True)],
        cancelled=False,
    )

    async def respond(args):
        return expected_result

    result = await run_tool_with_requests(tool, single_question_args, respond)

    assert result is not None
    assert result.answers[0].is_other is True
    assert result.answers[0].answer == "SQLite"


class TestToolUIDisplay:
    def test_get_call_display_single_question(self, single_question_args):
        event = ToolCallEvent(
            tool_name="ask_user_question",
            tool_class=AskUserQuestion,
            args=single_question_args,
            tool_call_id="123",
        )
        display = AskUserQuestion.get_call_display(event)
        assert "Which database?" in display.summary
        assert display.verb == "Asking"
        assert display.message == "Which database?"
        assert display.settled_verb == "Asked"
        assert display.settled_message == "Which database?"

    def test_get_call_display_multiple_questions(self, multi_question_args):
        event = ToolCallEvent(
            tool_name="ask_user_question",
            tool_class=AskUserQuestion,
            args=multi_question_args,
            tool_call_id="123",
        )
        display = AskUserQuestion.get_call_display(event)
        assert "2 questions" in display.summary
        assert display.verb == "Asking"
        assert display.message == "2 questions"
        assert display.settled_verb == "Asked"
        assert display.settled_message == "2 questions"

    def test_get_result_display_success(self):
        result = AskUserQuestionResult(
            answers=[Answer(question="Q?", answer="A", is_other=False)], cancelled=False
        )
        event = ToolResultEvent(
            tool_name="ask_user_question",
            tool_class=AskUserQuestion,
            result=result,
            tool_call_id="123",
        )
        display = AskUserQuestion.get_result_display(event)
        assert display.success is True
        assert "A" in display.message

    def test_get_result_display_cancelled(self):
        result = AskUserQuestionResult(answers=[], cancelled=True)
        event = ToolResultEvent(
            tool_name="ask_user_question",
            tool_class=AskUserQuestion,
            result=result,
            tool_call_id="123",
        )
        display = AskUserQuestion.get_result_display(event)
        assert display.success is False
        assert display.verb.lower() == "cancelled"

    def test_get_result_display_other(self):
        result = AskUserQuestionResult(
            answers=[Answer(question="Q?", answer="Custom", is_other=True)],
            cancelled=False,
        )
        event = ToolResultEvent(
            tool_name="ask_user_question",
            tool_class=AskUserQuestion,
            result=result,
            tool_call_id="123",
        )
        display = AskUserQuestion.get_result_display(event)
        assert display.success is True
        assert "(Other)" in display.message


def _defaultable_question(**overrides) -> Question:
    return Question(
        question="Which logging level?",
        header="Logging",
        options=[Choice(label="Info"), Choice(label="Debug")],
        kind=QuestionKind.PREFERENCE,
        unattended_default=UnattendedDefault(
            action=UnattendedDefaultAction.DEFER_AFFECTED_WORK,
            description="Defer enabling debug logging",
        ),
        **overrides,
    )


def _unanswered_result() -> UnansweredQuestions:
    return UnansweredQuestions(
        questions=[
            UnansweredQuestion(
                question_id="req-1:0",
                question="Which logging level?",
                header="Logging",
                kind=QuestionKind.PREFERENCE,
                unattended_default=UnattendedDefault(
                    action=UnattendedDefaultAction.DEFER_AFFECTED_WORK,
                    description="Defer enabling debug logging",
                ),
            )
        ],
        elapsed_seconds=61.0,
        timeout_seconds=60.0,
        continuation_guidance="Continue with the deferred default.",
    )


class TestUnattendedContract:
    """Plan U stage 1: the opt-in contract and typed dispositions."""

    def test_question_kind_defaults_to_requires_user(self):
        question = Question(
            question="Proceed?", options=[Choice(label="Yes"), Choice(label="No")]
        )
        assert question.kind is QuestionKind.REQUIRES_USER
        assert question.unattended_default is None

    def test_preserve_existing_choice_requires_an_option_label(self):
        with pytest.raises(ValidationError):
            Question(
                question="Proceed?",
                options=[Choice(label="Yes"), Choice(label="No")],
                kind=QuestionKind.PREFERENCE,
                unattended_default=UnattendedDefault(
                    action=UnattendedDefaultAction.PRESERVE_EXISTING_CHOICE,
                    description="Keep the current setting",
                    existing_choice="Maybe",
                ),
            )
        question = Question(
            question="Proceed?",
            options=[Choice(label="Yes"), Choice(label="No")],
            kind=QuestionKind.PREFERENCE,
            unattended_default=UnattendedDefault(
                action=UnattendedDefaultAction.PRESERVE_EXISTING_CHOICE,
                description="Keep the current setting",
                existing_choice="Yes",
            ),
        )
        default = question.unattended_default
        assert default is not None
        assert default.existing_choice == "Yes"

    def test_preserve_existing_choice_requires_existing_choice(self):
        with pytest.raises(ValidationError):
            UnattendedDefault(
                action=UnattendedDefaultAction.PRESERVE_EXISTING_CHOICE,
                description="Keep the current setting",
            )

    def test_dedicated_args_carry_timeout_override(self):
        args = AskUserQuestionArgs(
            questions=[_defaultable_question()], timeout_seconds=120.0
        )
        assert args.timeout_seconds == 120.0
        parameters = AskUserQuestion.get_parameters()
        assert "timeoutSeconds" in parameters["properties"]
        assert "kind" in parameters["$defs"]["UserQuestion"]["properties"]
        assert "unattendedDefault" in parameters["$defs"]["UserQuestion"]["properties"]
        # The generic runtime request stays free of the tool-only override.
        from chartreux.questions import UserQuestionRequest

        assert "timeout_seconds" not in UserQuestionRequest.model_fields

    def test_timeout_override_rejects_non_finite(self):
        with pytest.raises(ValidationError):
            AskUserQuestionArgs(
                questions=[_defaultable_question()], timeout_seconds=float("inf")
            )
        with pytest.raises(ValidationError):
            AskUserQuestionArgs(
                questions=[_defaultable_question()], timeout_seconds=0.0
            )

    def test_legacy_results_normalize_disposition(self):
        answered = UserQuestionResult(
            answers=[Answer(question="Q?", answer="A")], cancelled=False
        )
        assert answered.disposition == "answered"
        cancelled = UserQuestionResult(answers=[], cancelled=True)
        assert cancelled.disposition == "cancelled"

    def test_client_cannot_submit_unanswered_disposition(self):
        with pytest.raises(ValidationError):
            UserQuestionResult(answers=[], cancelled=False, disposition="unanswered")
        with pytest.raises(ValidationError):
            UserQuestionResult(answers=[], cancelled=True, disposition="answered")
        with pytest.raises(ValidationError):
            UserQuestionResult(answers=[], cancelled=False, disposition="cancelled")

    def test_unanswered_result_is_server_shaped(self):
        result = _unanswered_result()
        assert result.disposition == "unanswered"
        assert result.reason == "unanswered_timeout"
        assert result.answers == []
        assert result.cancelled is False
        assert result.questions[0].question_id == "req-1:0"
        assert result.elapsed_seconds == 61.0
        assert result.timeout_seconds == 60.0
        assert result.continuation_guidance
        # The plain client-submittable result type rejects the unanswered shape.
        with pytest.raises(ValidationError):
            UserQuestionResult.model_validate(result.model_dump(mode="json"))

    @pytest.mark.asyncio
    async def test_tool_stamps_model_question_origin(self, tool):
        requests = FakeInteractionRequests(user_input=lambda args: _coroutine_answer())
        ctx = InvokeContext(interaction_requests=requests, tool_call_id="123")
        args = AskUserQuestionArgs(questions=[_defaultable_question()])
        async for _ in tool.run(args, ctx):
            pass
        assert requests.origins == [UserInputOrigin.MODEL_QUESTION]

    @pytest.mark.asyncio
    async def test_tool_yields_unanswered_result(self, tool):
        unanswered = _unanswered_result()

        async def respond(args):
            return unanswered

        ctx = InvokeContext(
            interaction_requests=FakeInteractionRequests(user_input=respond),
            tool_call_id="123",
        )
        result: UserQuestionResult | None = None
        async for item in tool.run(
            AskUserQuestionArgs(questions=[_defaultable_question()]), ctx
        ):
            result = item
        assert isinstance(result, UnansweredQuestions)
        assert result is unanswered
        assert result.disposition == "unanswered"

    def test_get_result_display_unanswered(self):
        event = ToolResultEvent(
            tool_name="ask_user_question",
            tool_class=AskUserQuestion,
            result=_unanswered_result(),
            tool_call_id="123",
        )
        display = AskUserQuestion.get_result_display(event)
        assert display.success is False
        assert display.verb == "Unanswered"
        assert "Unanswered — deferred" in display.message
        assert "no permission granted" in display.message


async def _coroutine_answer():
    return UserQuestionResult(
        answers=[Answer(question="Which logging level?", answer="Info")]
    )
