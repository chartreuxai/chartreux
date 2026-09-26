from __future__ import annotations

import asyncio
import os
from pathlib import Path

from chartreux.core.tools.secret_redaction import scrub_child_env


async def spawn_shell_command(
    command: str, *, cwd: Path | None = None
) -> asyncio.subprocess.Process:
    env = _shell_environment()
    cwd = cwd or Path.cwd()
    return await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
        env=env,
        cwd=cwd,
        executable=os.environ.get("SHELL"),
        start_new_session=True,
    )


def _shell_environment() -> dict[str, str]:
    # Chartreux's credential variables never reach shell children unless the
    # user opted specific names back in via credential_env_passthrough.
    env = {
        **scrub_child_env(os.environ),
        "CI": "true",
        "NONINTERACTIVE": "1",
        "NO_TTY": "1",
    }
    # LC_ALL overrides every LC_* category, so a user-set LC_ALL=C (common in
    # CI/containers) would defeat LC_CTYPE below and yield non-UTF-8 output.
    env.pop("LC_ALL", None)
    return {
        **env,
        "TERM": "dumb",
        "DEBIAN_FRONTEND": "noninteractive",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "LESS": "-FX",
        "LC_CTYPE": "C.UTF-8",
    }


__all__ = ["spawn_shell_command"]
