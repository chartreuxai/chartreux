from __future__ import annotations

import asyncio
from collections.abc import Mapping
import os
from pathlib import Path

from chartreux.core.tools.secret_redaction import scrub_child_env
from chartreux.core.utils.async_subprocess import spawn_registered_process


async def spawn_shell_command(
    command: str,
    *,
    cwd: Path | None = None,
    merge_stderr: bool = False,
    extra_env: Mapping[str, str] | None = None,
) -> asyncio.subprocess.Process:
    env = _shell_environment(extra_env)
    cwd = cwd or Path.cwd()
    return await spawn_registered_process(
        asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
            if merge_stderr
            else asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            env=env,
            cwd=cwd,
            executable=os.environ.get("SHELL"),
            start_new_session=True,
        )
    )


def _shell_environment(extra_env: Mapping[str, str] | None = None) -> dict[str, str]:
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
    env = {
        **env,
        "TERM": "dumb",
        "DEBIAN_FRONTEND": "noninteractive",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "LESS": "-FX",
        "LC_CTYPE": "C.UTF-8",
    }
    if extra_env:
        # The project environment (the trusted folder's .env and
        # $CHARTREUX_HOME/.env) only fills names the harness did not set: an
        # explicit process/shell value wins over the .env files.
        env = {**extra_env, **env}
    return env


__all__ = ["spawn_shell_command"]
