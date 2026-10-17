"""Dangerous-command lifecycle — reprompt concern (stage 20c split target).

Denial behavior and re-prompt identity: refusals are remembered per call
identity (and only per shlex token tuple), and noninteractive sessions deny
promptly, naming the remediation. Moved here from
``test_dangerous_command_lifecycle.py`` in the 20c-ii-3 concern commit,
together with the single-concern ``_NeverCalledGrantPort`` helper. The
module re-registers the marks the population had in the original —
``pytestmark = pytest.mark.asyncio``; the original file defines no
module-local fixtures, so there is no fixture registration to carry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.config.types import ConfigSaveResult
from tests.agent_loop._lifecycle_test_support import (
    DECLINED_FEEDBACK,
    _authorities,
    _bash_args,
    _call,
    _config,
    _drive,
    _results,
    make_repo,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio


class _NeverCalledGrantPort:
    """Runtime marker only; a session that never approves never saves."""

    async def save_dangerous_command_grant(
        self, session_id: str, project: Path, operation_id: str, expected_revision: str
    ) -> ConfigSaveResult:
        raise AssertionError("A session without an approval must not save grants")


async def test_refusal_memory_does_not_block_unrelated_operations(
    tmp_path: Path,
) -> None:
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call("bash", _bash_args("git push"), "ref-1")])],
        [mock_llm_chunk(content="done")],
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git stash drop"), "ref-2")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
        # A different call identity is a different decision: it still prompts.
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


async def test_refusal_memory_matches_shlex_token_variants_only(tmp_path: Path) -> None:
    """Refusal identity is the whole-call shlex token tuple, nothing smarter.

    Whitespace and quoting variants of the denied command hit the same
    refusal record (no re-prompt); a semantically equivalent reformulation
    (-f vs --force) tokenizes differently and re-prompts.
    """
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call("bash", _bash_args("git push"), "canon-1")])],
        [mock_llm_chunk(content="done")],
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git  push"), "canon-2")]
            )
        ],
        [mock_llm_chunk(content="done")],
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args('git "push"'), "canon-3")]
            )
        ],
        [mock_llm_chunk(content="done")],
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git push -f"), "canon-4")]
            )
        ],
        [mock_llm_chunk(content="done")],
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git push --force"), "canon-5")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        # Deny the plain push: the refusal is remembered for its token tuple.
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
        # Whitespace and quoting variants are the same call identity: the
        # decided refusal answers, no re-prompt.
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        assert _results(events)[0].skipped
        assert _results(events)[0].skip_reason == DECLINED_FEEDBACK
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        assert _results(events)[0].skipped
        assert _results(events)[0].skip_reason == DECLINED_FEEDBACK
        # -f vs --force: semantically equivalent, different tokens, so the
        # reformulation is a new decision and prompts again.
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


async def test_noninteractive_consequential_denial_names_no_persistent_knob(
    tmp_path: Path,
) -> None:
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git stash drop"), "no-cap")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=False
    )
    try:
        agent.bind_dangerous_command_grant_port(_NeverCalledGrantPort())
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason is not None
        assert "cannot be pre-authorized" in results[0].skip_reason
        assert "interactive session" in results[0].skip_reason
        assert "dangerous_command_grants_by_project" not in results[0].skip_reason
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


async def test_noninteractive_mixed_call_names_only_the_persistable_part(
    tmp_path: Path,
) -> None:
    """A mixed call: the config knob pre-authorizes only part of the call."""
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[
                    _call(
                        "bash",
                        _bash_args("git push && git push --force"),
                        "mixed-no-cap",
                    )
                ]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=False
    )
    try:
        agent.bind_dangerous_command_grant_port(_NeverCalledGrantPort())
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        reason = results[0].skip_reason or ""
        # The persistable subset is named as partial pre-authorization only.
        assert "Add git.push to dangerous_command_grants_by_project" in reason
        assert "pre-authorize only part of this call" in reason
        assert "remaining operations (git.push.force)" in reason
        assert "cannot be pre-authorized" in reason
        assert "interactive session" in reason
        # The message must not claim the config edit authorizes the whole call.
        assert "to authorize it." not in reason
        assert _authorities(agent) == []
    finally:
        await agent.aclose()
