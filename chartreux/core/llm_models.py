from __future__ import annotations

from collections import OrderedDict
import copy
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)

from chartreux.user_content import UserDisplayContent, UserResource
from chartreux.utils.tool_presentation import (
    ToolCallPresentation,
    ToolResultPresentation,
)


class Backend(StrEnum):
    MISTRAL = auto()
    GENERIC = auto()


StrToolChoice = Literal["auto", "none", "any", "required"]


class AvailableFunction(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class AvailableTool(BaseModel):
    type: Literal["function"] = "function"
    function: AvailableFunction


class FunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCall(BaseModel):
    id: str | None = None
    index: int | None = None
    function: FunctionCall = Field(default_factory=FunctionCall)
    type: Literal["function"] = "function"
    presentation: ToolCallPresentation | None = None


def _content_before(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        parts: list[str] = []
        for p in v:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(v)


Content = Annotated[str, BeforeValidator(_content_before)]


class Role(StrEnum):
    system = auto()
    user = auto()
    assistant = auto()
    tool = auto()


class FileImageSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["file"] = "file"
    path: Path


class InlineImageSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["inline"] = "inline"
    # Raw base64-encoded bytes (no `data:` prefix). Used when the image has no
    # durable file on disk (session logging disabled): memory-only, never
    # persisted to a session transcript.
    data: str


class ImageAttachment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: Annotated[FileImageSource | InlineImageSource, Field(discriminator="kind")]
    alias: str
    mime_type: str


class PersistedToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output: dict[str, JsonValue]
    duration: float | None = Field(default=None, ge=0)
    cancelled: bool = False
    presentation: ToolResultPresentation | None = None


class ManualShellContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str
    command: str
    cwd: str
    stdout: str = ""
    stderr: str = ""
    output_text: str = ""
    exit_code: int
    timed_out: bool = False
    interrupted: bool = False
    duration_ms: float = Field(default=0.0, ge=0)
    created_at: int


class LLMMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Role
    content: Content | None = None
    images: list[ImageAttachment] | None = None
    injected: bool = False
    reasoning_content: Content | None = None
    reasoning_payloads: list[dict[str, Any]] | None = None
    reasoning_message_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_result: PersistedToolResult | None = None
    message_id: str | None = None
    user_display_content: UserDisplayContent | None = None
    input_text: str | None = None
    resources: list[UserResource] | None = None
    manual_shell: ManualShellContext | None = None
    context_boundary: Literal["compaction"] | None = None
    deployment_identity: dict[str, str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_any(cls, v: Any) -> dict[str, Any] | Any:
        if isinstance(v, dict):
            v.setdefault("content", "")
            v.setdefault("role", "assistant")
            if v.get("message_id") is None and v.get("role") != "tool":
                v["message_id"] = str(uuid4())
            if v.get("reasoning_message_id") is None and v.get("reasoning_content"):
                v["reasoning_message_id"] = str(uuid4())
            return v
        role = str(getattr(v, "role", "assistant"))
        reasoning_content = getattr(v, "reasoning_content", None)
        return {
            "role": role,
            "content": getattr(v, "content", ""),
            "reasoning_content": reasoning_content,
            "reasoning_payloads": getattr(v, "reasoning_payloads", None),
            "reasoning_message_id": getattr(v, "reasoning_message_id", None)
            or (str(uuid4()) if reasoning_content else None),
            "tool_calls": getattr(v, "tool_calls", None),
            "name": getattr(v, "name", None),
            "tool_call_id": getattr(v, "tool_call_id", None),
            "tool_result": getattr(v, "tool_result", None),
            "images": getattr(v, "images", None),
            "message_id": getattr(v, "message_id", None)
            or (str(uuid4()) if role != "tool" else None),
            "user_display_content": getattr(v, "user_display_content", None),
            "input_text": getattr(v, "input_text", None),
            "resources": getattr(v, "resources", None),
            "manual_shell": getattr(v, "manual_shell", None),
            "context_boundary": getattr(v, "context_boundary", None),
            "deployment_identity": getattr(v, "deployment_identity", None),
        }

    def __add__(self, other: LLMMessage) -> LLMMessage:
        """Careful: this is not commutative!"""
        if self.role != other.role:
            raise ValueError("Can't accumulate messages with different roles")

        if self.name != other.name:
            raise ValueError("Can't accumulate messages with different names")

        if self.tool_call_id != other.tool_call_id:
            raise ValueError("Can't accumulate messages with different tool_call_ids")

        content = (self.content or "") + (other.content or "")
        if not content:
            content = None

        reasoning_content = (self.reasoning_content or "") + (
            other.reasoning_content or ""
        )
        if not reasoning_content:
            reasoning_content = None

        reasoning_payloads = [
            *(self.reasoning_payloads or []),
            *(other.reasoning_payloads or []),
        ] or None

        tool_calls_map = OrderedDict[int, ToolCall]()
        for tool_calls in [self.tool_calls or [], other.tool_calls or []]:
            for tc in tool_calls:
                if tc.index is None:
                    raise ValueError("Tool call chunk missing index")
                if tc.index not in tool_calls_map:
                    tool_calls_map[tc.index] = copy.deepcopy(tc)
                else:
                    existing_name = tool_calls_map[tc.index].function.name
                    new_name = tc.function.name
                    if existing_name and new_name and existing_name != new_name:
                        raise ValueError(
                            "Can't accumulate messages with different tool call names"
                        )
                    if new_name and not existing_name:
                        tool_calls_map[tc.index].function.name = new_name
                    new_args = (tool_calls_map[tc.index].function.arguments or "") + (
                        tc.function.arguments or ""
                    )
                    tool_calls_map[tc.index].function.arguments = new_args

        return LLMMessage(
            role=self.role,
            content=content,
            images=self.images if self.images is not None else other.images,
            reasoning_content=reasoning_content,
            reasoning_payloads=reasoning_payloads,
            reasoning_message_id=self.reasoning_message_id
            or other.reasoning_message_id,
            tool_calls=list(tool_calls_map.values()) or None,
            name=self.name,
            tool_call_id=self.tool_call_id,
            tool_result=self.tool_result or other.tool_result,
            message_id=self.message_id,
            user_display_content=self.user_display_content
            if self.user_display_content is not None
            else other.user_display_content,
            input_text=(
                self.input_text if self.input_text is not None else other.input_text
            ),
            resources=self.resources if self.resources is not None else other.resources,
            context_boundary=self.context_boundary or other.context_boundary,
            deployment_identity=(self.deployment_identity or other.deployment_identity),
        )


class LLMUsage(BaseModel):
    model_config = ConfigDict(frozen=True)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Prompt tokens served from the provider cache; a subset of prompt_tokens.
    cached_tokens: int = 0

    def __add__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


class StopReason(StrEnum):
    REFUSAL = "refusal"


class StopInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    reason: str | None = None
    category: str | None = None
    explanation: str | None = None

    @property
    def is_refusal(self) -> bool:
        return self.reason == StopReason.REFUSAL


class LLMChunk(BaseModel):
    model_config = ConfigDict(frozen=True)
    message: LLMMessage
    usage: LLMUsage | None = None
    correlation_id: str | None = None
    stop: StopInfo | None = None

    def __add__(self, other: LLMChunk) -> LLMChunk:
        if self.usage is None and other.usage is None:
            new_usage = None
        else:
            new_usage = (self.usage or LLMUsage()) + (other.usage or LLMUsage())
        return LLMChunk(
            message=self.message + other.message,
            usage=new_usage,
            correlation_id=other.correlation_id or self.correlation_id,
            stop=other.stop or self.stop,
        )
