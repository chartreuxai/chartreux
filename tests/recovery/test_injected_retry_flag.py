"""WP0 F6 — the injected-flag conflation fix, four-path regression matrix.

The conflation (fixed in WP0): ``_turns.py`` computed
``user_initiated_retry = params.user_initiated_retry or params.injected``,
and ``_loop.py`` clears the committed base's shared availability cooldowns on
a user-initiated retry — so EVERY injected turn cleared cooldowns, including
the automatic incomplete-stream continuation and ACP injected turns. The fix
threads ``user_initiated_retry`` through the ACP producer (the ``/retry``
command marks its ``InjectedPrompt``; ``_TurnInput`` gains the field;
``act()`` passes it) and drops the conflation.

After the fix, manual ``/retry`` keeps its cooldown reset on BOTH surfaces
(TUI and ACP — the only user-action paths that reset), while the TUI
incomplete-stream continuation and any non-retry ACP injection never reset.

The four paths, each asserted MID-TURN or at admission with the shared
cooldown armed after a first successful turn (a successful turn's
``record_success`` would clear it, so the cooldown is armed between turns;
the reset fires at the next turn's admission, before its backend request):

1. TUI ``/retry`` — resets (the retry's backend request is admitted and the
   cooldown is gone while it is in flight).
2. ACP ``/retry`` — resets (same, through the ACP command producer).
3. TUI incomplete-stream continuation — never (the cooldown is armed during
   the first stream, after its admission; the automatic continuation is then
   blocked at admission: no second transport attempt).
4. ACP non-retry injected turn — never (driven directly through
   ``app_server.act(..., injected=True)``; the plan notes this path has no
   production producer today — ``InjectedPrompt`` is constructed only by the
   ``/retry`` handler — so the app-server injection boundary is driven
   directly).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from acp.schema import TextContentBlock
import pytest

from chartreux.app_server.models import TurnErrorCode
from chartreux.app_server.session import AppServerTurnError
from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import RecoverySectionConfig, RootRecoveryConfig
from chartreux.core.llm.exceptions import BackendError, PayloadSummary
from chartreux.utils.retry_prompt import build_retry_prompt
from tests.conftest import (
    build_test_agent_loop,
    build_test_chartreux_app,
    build_test_vibe_config,
    wait_until,
)
from tests.mock.utils import mock_llm_chunk
from tests.recovery.conftest import AcpRecoveryHarness


def _committed_key(loop: Any) -> tuple[str, str]:
    committed = loop.committed_model
    assert committed is not None
    return committed.base_model, committed.provider


def _rate_limit_error() -> BackendError:
    """A typed rate-limit fault: a retryable turn error (offers /retry)."""
    return BackendError(
        provider="mistral",
        endpoint="https://api.mistral.ai/v1/chat/completions",
        status=429,
        reason="rate limited",
        headers={"x-request-id": "req-wp0-f6"},
        body_text='{"error": {"code": "rate_limit_exceeded"}}',
        parsed_error='{"error": {"code": "rate_limit_exceeded"}}',
        model="zai-glm-5-3",
        payload_summary=PayloadSummary(
            model="zai-glm-5-3",
            message_count=1,
            approx_chars=10,
            temperature=0,
            has_tools=False,
            tool_choice=None,
        ),
    )


def _disabled_root_recovery_config() -> ChartreuxConfigSchema:
    """Explicitly-disabled ``[recovery.root]`` (Q, WP6 promotion ripple).

    WP6 promoted the enabled default, but the F6 path-3 matrix test
    characterizes the no-automatic-continuation surface — the surfaced error
    widget with no scheduled continuation — so the pre-promotion default
    (disabled) is pinned explicitly.
    """
    return build_test_vibe_config().model_copy(
        update={
            "recovery": RecoverySectionConfig(root=RootRecoveryConfig(enabled=False))
        }
    )


@pytest.mark.asyncio
async def test_f6_tui_retry_resets_shared_cooldown(controlled_transport) -> None:
    """Path 1 (TUI /retry): a user-initiated retry resets the cooldown.

    The first turn fails with a rate limit (a retryable turn error, so the
    TUI offers /retry). The failure arms the shared cooldown; the /retry turn
    resets it at admission, before its backend request starts.
    """
    backend = controlled_transport([_rate_limit_error()])
    loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    app = build_test_chartreux_app(agent_loop=loop)
    base, provider = _committed_key(loop)
    registry = loop.config_orchestrator.availability_registry

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        await wait_until(pilot, lambda: len(backend.requests_messages) == 1)
        await wait_until(pilot, lambda: not app._agent_job_active())

        # The rate-limited first turn armed the shared cooldown (and offered
        # /retry).
        assert registry.cooldown_until(base, provider) is not None

        # The /retry command exists only once the failed turn's retry
        # presentation is offered; the turn job can finish before the offer is
        # processed, so wait for the command gate before submitting it.
        assert await wait_until(pilot, lambda: app.commands.has_command("retry"))

        # Hold the retry turn's backend request in flight.
        started, release = backend.gate_attempt(1)
        backend.add_outcome([mock_llm_chunk(content="retried answer")])
        chat_input.post_message(ChatInputContainer.Submitted("/retry stay brief"))
        await asyncio.wait_for(started.wait(), timeout=5)

        # The user-initiated retry reset the base's shared cooldown at
        # admission, before the backend request started.
        assert registry.cooldown_until(base, provider) is None
        release.set()
        await wait_until(pilot, lambda: len(backend.requests_messages) == 2)
        await wait_until(pilot, lambda: not app._agent_job_active())

    retry_message = backend.requests_messages[-1][-1]
    assert retry_message.injected is True
    assert retry_message.content == build_retry_prompt("stay brief")


@pytest.mark.asyncio
async def test_f6_acp_retry_resets_shared_cooldown(
    acp_recovery_agent: AcpRecoveryHarness,
) -> None:
    """Path 2 (ACP /retry): the command's InjectedPrompt resets the cooldown."""
    harness = acp_recovery_agent
    loop = harness.loop
    backend = harness.backend
    base, provider = _committed_key(loop)
    registry = loop.config_orchestrator.availability_registry

    created = await harness.agent.new_session(cwd=str(Path.cwd()), mcp_servers=[])
    await harness.agent.prompt(
        session_id=created.session_id,
        prompt=[TextContentBlock(type="text", text="Hello")],
    )
    assert len(backend.requests_messages) == 1

    # Arm the shared cooldown after the successful first turn.
    registry.record_failure(base, provider)
    assert registry.cooldown_until(base, provider) is not None

    # Hold the retry turn's backend request in flight.
    started, release = backend.gate_attempt(1)
    backend.add_outcome([mock_llm_chunk(content="Continued response.")])
    prompt_task = asyncio.create_task(
        harness.agent.prompt(
            session_id=created.session_id,
            prompt=[TextContentBlock(type="text", text="/retry stay brief")],
        )
    )
    await asyncio.wait_for(started.wait(), timeout=5)

    # The ACP /retry is user-initiated: its cooldown reset survives the fix
    # (it no longer depends on the injected-flag conflation).
    assert registry.cooldown_until(base, provider) is None
    release.set()
    response = await asyncio.wait_for(prompt_task, timeout=5)
    assert response.stop_reason == "end_turn"
    assert len(backend.requests_messages) == 2
    retry_message = backend.requests_messages[-1][-1]
    assert retry_message.injected is True
    assert retry_message.content == build_retry_prompt("stay brief")


@pytest.mark.asyncio
async def test_f6_tui_incomplete_stream_continuation_never_resets(
    controlled_transport,
) -> None:
    """Path 3 (TUI incomplete-stream): the automatic continuation never resets.

    The first stream is incomplete (no stop reason). The cooldown is armed
    during that stream — after the first turn's admission has already passed —
    so the automatic continuation's admission sees the cooldown. With the fix,
    the continuation does NOT reset it and is blocked at admission: no second
    transport attempt, and the surfaced error is the cooldown block.

    Q (WP6 promotion ripple): root recovery is pinned OFF here — with no
    automatic continuation scheduled at all, the incomplete stream surfaces
    its error directly; the never-resets property is unchanged (no second
    transport attempt, the cooldown stays armed).
    """
    backend = controlled_transport([
        [mock_llm_chunk(content="partial answer", stop_reason=None)],
        [mock_llm_chunk(content=" recovered")],
    ])
    loop = build_test_agent_loop(
        config=_disabled_root_recovery_config(), backend=backend, enable_streaming=True
    )
    app = build_test_chartreux_app(agent_loop=loop)
    base, provider = _committed_key(loop)
    registry = loop.config_orchestrator.availability_registry

    def arm_on_first_attempt(index: int) -> None:
        if index == 0:
            registry.record_failure(base, provider)

    backend.attempt_hook = arm_on_first_attempt

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        await wait_until(pilot, lambda: len(backend.requests_messages) == 1)
        await wait_until(pilot, lambda: not app._agent_job_active())

        # The automatic incomplete-stream continuation never resets the
        # cooldown: it is blocked at admission, so no second transport
        # attempt exists and the cooldown is still armed.
        assert len(backend.requests_messages) == 1
        assert registry.cooldown_until(base, provider) is not None
        # The blocked continuation surfaces as an error, not a silent retry
        # (wait for the widget: the job can finish before it is mounted).
        assert await wait_until(pilot, lambda: len(app.query(ErrorMessage)) == 1)


@pytest.mark.asyncio
async def test_f6_acp_non_retry_injected_turn_never_resets(
    acp_recovery_agent: AcpRecoveryHarness,
) -> None:
    """Path 4 (ACP non-retry injected): an injected turn never resets.

    No production producer exists for a non-retry ACP injection (the plan's
    round-3 dispatch note), so the app-server injection boundary is driven
    directly: ``act(..., injected=True)`` with ``user_initiated_retry`` unset.
    With the cooldown armed, the injected turn must NOT reset it and is
    blocked at admission: no second transport attempt.
    """
    harness = acp_recovery_agent
    loop = harness.loop
    backend = harness.backend
    base, provider = _committed_key(loop)
    registry = loop.config_orchestrator.availability_registry

    created = await harness.agent.new_session(cwd=str(Path.cwd()), mcp_servers=[])
    await harness.agent.prompt(
        session_id=created.session_id,
        prompt=[TextContentBlock(type="text", text="Hello")],
    )
    assert len(backend.requests_messages) == 1

    # Arm the shared cooldown after the successful first turn.
    registry.record_failure(base, provider)
    assert registry.cooldown_until(base, provider) is not None

    # Drive the injection boundary directly (no production producer).
    session = harness.agent.sessions[created.session_id]
    with pytest.raises(AppServerTurnError) as exc_info:
        async for _event in session.app_server.act(
            "automatic continuation", injected=True
        ):
            pass

    # The injected turn never reset the cooldown: it is blocked at admission.
    assert exc_info.value.error.code == TurnErrorCode.ALL_DEPLOYMENTS_UNAVAILABLE
    assert len(backend.requests_messages) == 1
    assert registry.cooldown_until(base, provider) is not None
