from __future__ import annotations

import asyncio
import logging
import os
import signal

logger = logging.getLogger(__name__)


async def kill_async_subprocess(
    proc: asyncio.subprocess.Process, *, kill_process_group: bool = True
) -> None:
    """Force-terminate an asyncio child process and wait until it exits.

    With ``kill_process_group=True`` (default), the child is expected to be
    isolated in its own process group (for example with
    ``start_new_session=True``). Pass ``kill_process_group=False`` to kill
    only the child process; group isolation is not detected automatically.
    """
    if proc.returncode is not None:
        return

    try:
        if not kill_process_group:
            proc.kill()
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            except Exception:
                logger.debug(
                    "Unexpected error killing process group for pid %s",
                    proc.pid,
                    exc_info=True,
                )

        await proc.wait()
    except (ProcessLookupError, PermissionError, OSError):
        pass
