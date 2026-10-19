from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

__all__ = [
    "QuestionChoice",
    "QuestionKind",
    "UnansweredQuestion",
    "UnansweredQuestions",
    "UnattendedDefault",
    "UnattendedDefaultAction",
    "UserAnswer",
    "UserQuestion",
    "UserQuestionRequest",
    "UserQuestionResult",
]


class QuestionModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel, extra="forbid", populate_by_name=True
    )


class QuestionChoice(QuestionModel):
    label: str = Field(description="Short label for the choice (1-5 words)")
    description: str = Field(
        default="", description="Optional explanation of this choice"
    )


class QuestionKind(StrEnum):
    """What a user question needs from the user (plan U/OD1).

    ``requires_user`` is the default and blocks unattended expiry for the
    whole batch; ``informational`` and ``preference`` questions may declare
    an ``unattended_default`` and become eligible for expiry.
    """

    INFORMATIONAL = "informational"
    PREFERENCE = "preference"
    REQUIRES_USER = "requires_user"


class UnattendedDefaultAction(StrEnum):
    """What expiry does with the affected work (plan U/OD1).

    Both actions defer: neither authorizes anything new, and neither
    fabricates a user answer.
    """

    DEFER_AFFECTED_WORK = "defer_affected_work"
    PRESERVE_EXISTING_CHOICE = "preserve_existing_choice"


class UnattendedDefault(QuestionModel):
    """The declared unattended outcome for one question (plan U/OD1).

    Expiry means "no user decision received — defer affected work", never
    "the declared choice is approved". ``preserve_existing_choice`` names an
    already-existing choice to keep; it never introduces a new one.
    """

    action: UnattendedDefaultAction = Field(
        description="What to do with the affected work when no answer arrives"
    )
    description: str = Field(
        description="Plain-language description of the deferred outcome", min_length=1
    )
    existing_choice: str | None = Field(
        default=None,
        description=(
            "The existing option label to preserve; required when action is "
            "preserve_existing_choice"
        ),
    )

    @model_validator(mode="after")
    def _require_existing_choice(self) -> UnattendedDefault:
        if (
            self.action is UnattendedDefaultAction.PRESERVE_EXISTING_CHOICE
            and not self.existing_choice
        ):
            raise ValueError(
                "unattended_default.existing_choice is required when action is "
                "preserve_existing_choice"
            )
        return self


class UserQuestion(QuestionModel):
    question: str = Field(description="The question text")
    header: str = Field(
        default="",
        description="Short chip/tag label for the question (max 20 chars)",
        max_length=20,
    )
    options: list[QuestionChoice] = Field(
        description="Available options (2-4 recommended, not including 'Other'). An 'Other' option for free text is automatically added.",
        min_length=2,
    )
    multi_select: bool = Field(
        default=False, description="If true, user can select multiple options"
    )
    hide_other: bool = Field(
        default=False, description="If true, hide the 'Other' free text option"
    )
    kind: QuestionKind = Field(
        default=QuestionKind.REQUIRES_USER,
        description=(
            "informational/preference questions may declare an unattended_default "
            "and expire when unattended; requires_user (the default) always blocks "
            "expiry for the whole batch"
        ),
    )
    unattended_default: UnattendedDefault | None = Field(
        default=None,
        description=(
            "Declared unattended outcome. Missing means the question never "
            "expires. Expiry defers work; it never approves a choice."
        ),
    )

    @model_validator(mode="after")
    def _validate_unattended_default(self) -> UserQuestion:
        default = self.unattended_default
        if (
            default is not None
            and default.action is UnattendedDefaultAction.PRESERVE_EXISTING_CHOICE
        ):
            labels = {option.label for option in self.options}
            if default.existing_choice not in labels:
                raise ValueError(
                    "unattended_default.existing_choice must name one of this "
                    "question's option labels"
                )
        return self


class UserQuestionRequest(QuestionModel):
    questions: list[UserQuestion] = Field(
        description="Questions to ask (1-4 recommended). Displayed as tabs if multiple.",
        min_length=1,
    )
    footer_note: str | None = Field(
        default=None,
        description="Optional subtle note displayed at the bottom of the question widget.",
    )


class UserAnswer(QuestionModel):
    question: str = Field(description="The original question")
    answer: str = Field(description="The user's answer")
    is_other: bool = Field(
        default=False, description="True if user typed a custom answer via 'Other'"
    )


class UserQuestionResult(QuestionModel):
    answers: list[UserAnswer] = Field(description="List of answers")
    cancelled: bool = Field(
        default=False,
        description="True if the user cancelled without answering the questions.",
    )
    disposition: Literal["answered", "cancelled", "unanswered"] | None = Field(
        default=None,
        description=(
            "Typed disposition. Legacy results normalize from answers/cancelled; "
            "'unanswered' is server-owned and rejected on client submission."
        ),
    )

    @model_validator(mode="after")
    def _normalize_disposition(self) -> UserQuestionResult:
        # The client-submittable boundary: a plain UserQuestionResult can
        # never carry the server-owned unanswered disposition (the expiry
        # result is the UnansweredQuestions subclass, which overrides this
        # validator's guard via its exact type).
        if type(self) is UserQuestionResult:
            if self.disposition == "unanswered":
                raise ValueError(
                    "disposition 'unanswered' is server-owned; clients cannot submit it"
                )
            if self.disposition is None:
                self.disposition = "cancelled" if self.cancelled else "answered"
            elif self.disposition == "cancelled" and not self.cancelled:
                raise ValueError("disposition 'cancelled' requires cancelled=true")
            elif self.disposition == "answered" and self.cancelled:
                raise ValueError("disposition 'answered' contradicts cancelled=true")
        return self


class UnansweredQuestion(QuestionModel):
    """One echoed question with its stable ID and declared default (plan U)."""

    question_id: str = Field(description="Stable question ID within the request")
    question: str = Field(description="The original question text")
    header: str = Field(default="", description="The original question header")
    kind: QuestionKind = Field(description="The question's declared kind")
    unattended_default: UnattendedDefault = Field(
        description="The declared unattended default"
    )


class UnansweredQuestions(UserQuestionResult):
    """The server-owned unattended-expiry result (plan U/OD1).

    Produced only by the app server's callback-record timer when an eligible
    question receives no user answer in time. It carries no fabricated
    UserAnswer: ``answers`` stays empty and ``cancelled`` stays false, and the
    typed disposition is ``unanswered`` with reason ``unanswered_timeout``.
    Clients cannot submit this disposition: the callback wire type is the
    plain UserQuestionResult, which rejects both the extra fields and the
    'unanswered' disposition.
    """

    # Server-owned narrowing of the base's mutable field; pydantic enforces
    # the Literal, so this result always carries disposition 'unanswered'.
    disposition: Literal["unanswered"] = "unanswered"  # pyright: ignore[reportIncompatibleVariableOverride]
    answers: list[UserAnswer] = Field(default_factory=list)
    cancelled: bool = False
    questions: list[UnansweredQuestion] = Field(
        description="Echoed questions with stable IDs and declared defaults",
        min_length=1,
    )
    elapsed_seconds: float = Field(
        ge=0.0, description="Monotonic seconds the question waited unanswered"
    )
    timeout_seconds: float = Field(
        gt=0.0, description="The effective timeout the question expired under"
    )
    reason: Literal["unanswered_timeout"] = "unanswered_timeout"
    continuation_guidance: str = Field(
        description="Model-facing guidance for continuing after expiry"
    )
