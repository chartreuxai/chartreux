"""Shared fixtures for the app_server suite.

S2 round 2 (unvalidated ``user:``-qualified identities; design criterion 2):
a loop binds and validates its identity at construction, and a
``user:``-qualified custom identity requires its ``<name>.md`` prompt resource
in the user prompts directories (``CHARTREUX_HOME/prompts``). The custom
identities exercised across this suite are test fixtures, not real user
prompts, so the suite installs a prompt resource for each of them into the
per-test ``config_dir``. The resource's existence is the whole contract: a
child's system prompt comes from its identity's prompt file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Custom identity names used across the app_server suite. A bare custom name
# resolves as a ``user:<name>``-qualified custom identity (design decision 8),
# which validates its prompt resource at loop construction.
_CUSTOM_PROFILE_NAMES = (
    "dynamic",
    "final-response",
    "interactive",
    "my-scout",
    "neutral-test-agent",
    "noisy",
    "persistent",
    "scout",
    "wp3-agent",
)


@pytest.fixture(autouse=True)
def _custom_profile_identity_prompts(config_dir: Path) -> None:
    prompts = config_dir / "prompts"
    prompts.mkdir(exist_ok=True)
    for name in _CUSTOM_PROFILE_NAMES:
        (prompts / f"{name}.md").write_text(
            f"# {name}\n\nTest-only custom identity prompt.\n", encoding="utf-8"
        )
