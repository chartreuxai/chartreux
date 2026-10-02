"""Opt-in, inference-only diagnostics for one concrete provider deployment."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import json
import logging
import math
import struct
import time
from typing import Literal
import zlib

from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.llm.backend.factory import create_backend
from chartreux.core.llm.thinking_levels import get_thinking_levels
from chartreux.core.llm_models import (
    AvailableFunction,
    AvailableTool,
    ImageAttachment,
    InlineImageSource,
    LLMChunk,
    LLMMessage,
    Role,
)
from chartreux.core.utils import RequestRetryBudget
from chartreux.utils.api_keys import resolve_api_key_with_origin

SmokeStatus = Literal["pass", "fail", "unsupported", "unverified"]
Capability = Literal["tool", "thinking", "image"]

_active_probes = 0
_logging_state: tuple[int, int, list[logging.Handler]] | None = None


@dataclass(frozen=True, slots=True)
class CapabilityResult:
    status: SmokeStatus
    reason: str


@dataclass(frozen=True, slots=True)
class ProviderSmokeResult:
    provider: str
    model: str
    alias: str
    tool: CapabilityResult
    thinking: CapabilityResult
    image: CapabilityResult


def _image_fixture() -> ImageAttachment:
    """Generate a 32x32 solid red RGB PNG entirely in memory."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + kind
            + data
            + struct.pack("!I", zlib.crc32(kind + data))
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!2I5B", 32, 32, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\x00\x00" * 32) * 32))
        + chunk(b"IEND", b"")
    )
    return ImageAttachment(
        source=InlineImageSource(data=base64.b64encode(png).decode("ascii")),
        alias="smoke-fixture",
        mime_type="image/png",
    )


def _tool_fixture() -> AvailableTool:
    return AvailableTool(
        function=AvailableFunction(
            name="chartreux_smoke_echo",
            description="Synthetic diagnostic tool; never executed.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string", "enum": ["smoke-ok"]}},
                "required": ["value"],
                "additionalProperties": False,
            },
        )
    )


def _tool_verdict(message: LLMMessage) -> CapabilityResult:
    calls = message.tool_calls or []
    if not calls:
        return CapabilityResult("fail", "missing_tool_call")
    if len(calls) != 1 or calls[0].function.name != "chartreux_smoke_echo":
        return CapabilityResult("fail", "wrong_tool_name_or_count")
    try:
        arguments = json.loads(calls[0].function.arguments or "")
    except (ValueError, TypeError):
        return CapabilityResult("fail", "malformed_tool_arguments")
    if arguments != {"value": "smoke-ok"}:
        return CapabilityResult("fail", "unexpected_tool_arguments")
    return CapabilityResult("pass", "expected_tool_call")


def _verdict(capability: Capability, response: LLMChunk) -> CapabilityResult:
    if response.stop and response.stop.reason in {
        "length",
        "max_tokens",
        "max_output_tokens",
        "model_length",
        "incomplete",
    }:
        return CapabilityResult("fail", "output_truncated")
    message = response.message
    if capability == "tool":
        return _tool_verdict(message)
    if capability == "image":
        if (message.content or "").strip().casefold() != "red":
            return CapabilityResult("fail", "wrong_image_answer")
        return CapabilityResult("pass", "expected_image_answer")
    # Opaque/encrypted payloads alone do not establish observable reasoning.
    if (message.reasoning_content or "").strip():
        return CapabilityResult("pass", "observable_reasoning")
    return CapabilityResult("unverified", "accepted_without_observable_reasoning")


def _result(
    model: ModelConfig,
    provider: ProviderConfig,
    results: dict[Capability, CapabilityResult],
) -> ProviderSmokeResult:
    return ProviderSmokeResult(
        provider.name,
        model.name,
        model.alias,
        results["tool"],
        results["thinking"],
        results["image"],
    )


async def probe_provider_smoke(
    *,
    model: ModelConfig,
    provider: ProviderConfig,
    deadline_seconds: float = 30.0,
    max_tokens: int = 2048,
    enable_system_trust_store: bool = False,
) -> ProviderSmokeResult:
    """Run an isolated diagnostic with logging muted, including SDK debug bodies.

    Logging suppression is process-wide for this non-interactive diagnostic. The
    original root logging state is restored when the last overlapping probe exits,
    even on cancellation. Entry and exit do not yield to other async tasks.
    """
    global _active_probes, _logging_state
    if _active_probes == 0:
        _logging_state = (
            logging.root.manager.disable,
            logging.root.level,
            logging.root.handlers.copy(),
        )
        logging.disable(max(_logging_state[0], logging.CRITICAL))
    _active_probes += 1
    try:
        return await _probe_provider_smoke(
            model=model,
            provider=provider,
            deadline_seconds=deadline_seconds,
            max_tokens=max_tokens,
            enable_system_trust_store=enable_system_trust_store,
        )
    finally:
        _active_probes -= 1
        if _active_probes == 0 and _logging_state is not None:
            previous, level, handlers = _logging_state
            logging.root.handlers[:] = handlers
            logging.root.setLevel(level)
            logging.disable(previous)
            _logging_state = None


async def _probe_provider_smoke(  # noqa: PLR0912, PLR0915 - independent capability checks and failure stages
    *,
    model: ModelConfig,
    provider: ProviderConfig,
    deadline_seconds: float,
    max_tokens: int,
    enable_system_trust_store: bool,
) -> ProviderSmokeResult:
    """Probe a deployment directly, without retries, failover or persistence.

    The deadline includes synchronous credential resolution elapsed time; that
    lookup itself cannot be interrupted. Backend cleanup runs even after timeout
    or cancellation. Reasons are fixed diagnostic codes, never provider bodies.
    Invalid bounds/target raise ValueError; caller cancellation is propagated.
    """
    if not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be finite and positive")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer")
    if model.provider != provider.name:
        raise ValueError("model and provider must identify the same deployment")

    levels = (
        get_thinking_levels(str(provider.backend), provider.api_style, model.name) or {}
    )
    allowed = model.supported_thinking_levels
    thinking = next(
        (
            level
            for level in ("low", "medium", "high", "max")
            if level in levels
            and levels[level] not in {None, "none"}
            and (allowed is None or level in allowed)
        ),
        None,
    )
    pending = CapabilityResult("fail", "overall_deadline_exceeded")
    results: dict[Capability, CapabilityResult] = {
        "tool": pending,
        "thinking": pending
        if thinking
        else CapabilityResult("unsupported", "no_non_off_thinking_level"),
        "image": pending
        if model.supports_images
        else CapabilityResult("unsupported", "deployment_has_no_image_support"),
    }

    def fail_pending(reason: str) -> None:
        for capability in results:
            if results[capability] is pending:
                results[capability] = CapabilityResult("fail", reason)

    start = time.monotonic()
    try:
        credential = resolve_api_key_with_origin(provider.api_key_env_var)
    except Exception:
        fail_pending("credential_resolution_failed")
        return _result(model, provider, results)
    if credential is None and provider.api_key_env_var:
        fail_pending("credential_missing")
        return _result(model, provider, results)
    remaining = deadline_seconds - (time.monotonic() - start)
    if remaining <= 0:
        return _result(model, provider, results)

    try:
        backend = create_backend(
            provider=provider,
            resolved_credential=credential,
            retry_budget=RequestRetryBudget(max_elapsed_time=0),
            timeout=remaining,
            connect_timeout=remaining,
            write_timeout=remaining,
            pool_timeout=remaining,
            enable_system_trust_store=enable_system_trust_store,
        )
    except Exception:
        fail_pending("backend_initialization_failed")
        return _result(model, provider, results)
    try:
        async with asyncio.timeout(
            max(0, deadline_seconds - (time.monotonic() - start))
        ):
            baseline = next(
                (
                    level
                    # Wire-off aliases (e.g. Mistral "low") are still valid
                    # baselines even though they cannot verify thinking.
                    for level in ("off", thinking, model.thinking, *levels)
                    if level in levels and (allowed is None or level in allowed)
                ),
                None,
            )
            for capability in ("tool", "thinking", "image"):
                if results[capability].status == "unsupported":
                    continue
                tool = _tool_fixture() if capability == "tool" else None
                prompt = {
                    "tool": 'Call chartreux_smoke_echo exactly once with value "smoke-ok".',
                    "thinking": "Think through 17 * 19, then give the answer briefly.",
                    "image": "What color fills this image? Reply with exactly one color word.",
                }[capability]
                request_model = model.model_copy(
                    update={
                        "thinking": thinking if capability == "thinking" else baseline
                    }
                )
                try:
                    response = await backend.complete(
                        model=request_model,
                        messages=[
                            LLMMessage(
                                role=Role.user,
                                content=prompt,
                                images=[_image_fixture()]
                                if capability == "image"
                                else None,
                            )
                        ],
                        temperature=0.0,
                        tools=[tool] if tool else None,
                        tool_choice=tool,
                        max_tokens=max_tokens,
                        extra_headers=None,
                    )
                    results[capability] = _verdict(capability, response)
                except Exception:
                    results[capability] = CapabilityResult("fail", "request_failed")
    except TimeoutError:
        pass
    except Exception:
        fail_pending("probe_failed")
    finally:
        try:
            await backend.__aexit__(None, None, None)
        except Exception:
            for capability in results:
                if results[capability].status not in {"unsupported", "pass"}:
                    results[capability] = CapabilityResult(
                        "fail", "backend_cleanup_failed"
                    )
    return _result(model, provider, results)
