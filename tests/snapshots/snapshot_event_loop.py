from __future__ import annotations

import asyncio
import contextvars


def install_snapshot_wake() -> None:
    """Keep snapshot event loops waking while executor work is torn down."""
    loop = asyncio.get_running_loop()
    pulse_attr = "_chartreux_snapshot_wake_handle"
    pulse_context = contextvars.Context()

    def callback() -> None:
        setattr(
            loop, pulse_attr, loop.call_later(0.05, callback, context=pulse_context)
        )

    # Thread-future completions may not wake the selector in restricted test
    # environments; keep one pulse until loop close so executor teardown can
    # progress.
    if getattr(loop, pulse_attr, None) is None:
        setattr(
            loop, pulse_attr, loop.call_later(0.05, callback, context=pulse_context)
        )
