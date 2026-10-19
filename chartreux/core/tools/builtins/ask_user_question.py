from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import cast

from pydantic import Field

from chartreux.core.events import ToolResultEvent, UserInputOrigin
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from chartreux.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from chartreux.questions import (
    UnansweredQuestions,
    UserQuestionRequest,
    UserQuestionResult,
)
from chartreux.utils.tool_presentation import ToolEffectKind

# The result covers all three typed dispositions (answered | cancelled |
# unanswered); the unanswered disposition is the server-owned
# UnansweredQuestions subclass (plan U/OD1).
AskUserQuestionResult = UserQuestionResult


class AskUserQuestionArgs(UserQuestionRequest):
    """Dedicated ask_user_question arguments (plan U/OD1).

    Not an alias of the generic runtime request: the tool contract owns the
    request-level unattended-timeout override. The override must fit the
    configured [user_questions] range; it is rejected, never clamped.
    """

    timeout_seconds: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "Optional per-request unattended timeout in seconds. Must fit the "
            "configured [user_questions] range; out-of-range values are "
            "rejected. Only questions with kind informational/preference and "
            "an unattended_default may expire."
        ),
    )


class AskUserQuestionConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class AskUserQuestion(
    BaseTool[
        AskUserQuestionArgs, AskUserQuestionResult, AskUserQuestionConfig, BaseToolState
    ],
    ToolUIData[AskUserQuestionArgs, AskUserQuestionResult],
):
    effect_kind = ToolEffectKind.USER_QUESTION

    @classmethod
    def format_call_display(cls, args: AskUserQuestionArgs) -> ToolCallDisplay:
        count = len(args.questions)
        if count == 1:
            question = args.questions[0].question
            return ToolCallDisplay(
                summary=f"Asking: {question}",
                verb="Asking",
                message=question,
                settled_verb="Asked",
                settled_message=question,
            )
        return ToolCallDisplay(
            summary=f"Asking {count} questions",
            verb="Asking",
            message=f"{count} questions",
            settled_verb="Asked",
            settled_message=f"{count} questions",
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if event.error:
            return ToolResultDisplay(success=False, message=event.error)

        result = event.result
        if isinstance(result, UnansweredQuestions):
            return ToolResultDisplay(
                success=False,
                verb="Unanswered",
                message=(
                    f"Unanswered — deferred (no answer within "
                    f"{result.timeout_seconds:g}s); no permission granted"
                ),
            )

        if not isinstance(result, AskUserQuestionResult):
            return ToolResultDisplay(success=True, verb="Answered", message="")

        if result.cancelled:
            return ToolResultDisplay(success=False, verb="Cancelled", message="by user")

        parts = [
            f'"{a.question}" → {"(Other) " if a.is_other else ""}{a.answer}'
            for a in result.answers
        ]
        return ToolResultDisplay(
            success=True, verb="Answered", message=" · ".join(parts)
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Waiting for user input"

    async def run(
        self, args: AskUserQuestionArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[AskUserQuestionResult, None]:
        if ctx is None or ctx.interaction_requests is None:
            raise ToolError(
                "User input not available. This tool requires an interactive UI."
            )

        result = await ctx.interaction_requests.request_user_input(
            args,
            ctx.tool_call_id,
            # Model-originated question: the only origin eligible for
            # unattended expiry (plan U/OD1).
            origin=UserInputOrigin.MODEL_QUESTION,
        )
        yield cast(AskUserQuestionResult, result)
