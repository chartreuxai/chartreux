"""Utilities package. Re-exports all public and test-used symbols from submodules.

Import read_safe / read_safe_async / decode_safe (returns ReadSafeResult) from chartreux.utils.io and create_slug from
chartreux.core.utils.slug when needed to avoid circular imports with config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chartreux.core.llm.failures import RequestRetryBudget
    from chartreux.core.utils.async_subprocess import kill_async_subprocess
    from chartreux.core.utils.concurrency import (
        AsyncExecutor,
        ConversationLimitException,
        run_sync,
    )
    from chartreux.core.utils.matching import name_matches
    from chartreux.core.utils.merge import MergeConflictError, MergeStrategy
    from chartreux.core.utils.pacing import AdaptivePacer
    from chartreux.core.utils.retry import (
        RetryCategory,
        RetryObserver,
        RetryReason,
        StreamHTTPError,
        async_generator_retry,
        async_retry,
    )
    from chartreux.core.utils.sse import iter_sse_lines
    from chartreux.core.utils.tags import (
        CANCELLATION_TAG,
        KNOWN_TAGS,
        TOOL_ERROR_TAG,
        VIBE_STOP_EVENT_TAG,
        CancellationReason,
        TaggedText,
        get_user_cancellation_message,
        is_user_cancellation_event,
    )
    from chartreux.core.utils.time import utc_now
    from chartreux.utils.paths import is_dangerous_directory
    from chartreux.utils.platform import (
        get_platform_display_name,
        get_platform_id,
        get_platform_version,
    )

__all__ = [
    "CANCELLATION_TAG",
    "KNOWN_TAGS",
    "TOOL_ERROR_TAG",
    "VIBE_STOP_EVENT_TAG",
    "AdaptivePacer",
    "AsyncExecutor",
    "CancellationReason",
    "ConversationLimitException",
    "MergeConflictError",
    "MergeStrategy",
    "RequestRetryBudget",
    "RetryCategory",
    "RetryObserver",
    "RetryReason",
    "StreamHTTPError",
    "TaggedText",
    "async_generator_retry",
    "async_retry",
    "get_platform_display_name",
    "get_platform_id",
    "get_platform_version",
    "get_user_cancellation_message",
    "is_dangerous_directory",
    "is_user_cancellation_event",
    "iter_sse_lines",
    "kill_async_subprocess",
    "name_matches",
    "run_sync",
    "utc_now",
]

_MAPPING: dict[str, tuple[str, str]] = {
    "kill_async_subprocess": (
        "chartreux.core.utils.async_subprocess",
        "kill_async_subprocess",
    ),
    "AsyncExecutor": ("chartreux.core.utils.concurrency", "AsyncExecutor"),
    "ConversationLimitException": (
        "chartreux.core.utils.concurrency",
        "ConversationLimitException",
    ),
    "run_sync": ("chartreux.core.utils.concurrency", "run_sync"),
    "name_matches": ("chartreux.core.utils.matching", "name_matches"),
    "MergeConflictError": ("chartreux.core.utils.merge", "MergeConflictError"),
    "MergeStrategy": ("chartreux.core.utils.merge", "MergeStrategy"),
    "AdaptivePacer": ("chartreux.core.utils.pacing", "AdaptivePacer"),
    "RequestRetryBudget": ("chartreux.core.llm.failures", "RequestRetryBudget"),
    "RetryCategory": ("chartreux.core.utils.retry", "RetryCategory"),
    "RetryObserver": ("chartreux.core.utils.retry", "RetryObserver"),
    "RetryReason": ("chartreux.core.utils.retry", "RetryReason"),
    "StreamHTTPError": ("chartreux.core.utils.retry", "StreamHTTPError"),
    "async_generator_retry": ("chartreux.core.utils.retry", "async_generator_retry"),
    "async_retry": ("chartreux.core.utils.retry", "async_retry"),
    "iter_sse_lines": ("chartreux.core.utils.sse", "iter_sse_lines"),
    "CANCELLATION_TAG": ("chartreux.core.utils.tags", "CANCELLATION_TAG"),
    "KNOWN_TAGS": ("chartreux.core.utils.tags", "KNOWN_TAGS"),
    "TOOL_ERROR_TAG": ("chartreux.core.utils.tags", "TOOL_ERROR_TAG"),
    "VIBE_STOP_EVENT_TAG": ("chartreux.core.utils.tags", "VIBE_STOP_EVENT_TAG"),
    "CancellationReason": ("chartreux.core.utils.tags", "CancellationReason"),
    "TaggedText": ("chartreux.core.utils.tags", "TaggedText"),
    "get_user_cancellation_message": (
        "chartreux.core.utils.tags",
        "get_user_cancellation_message",
    ),
    "is_user_cancellation_event": (
        "chartreux.core.utils.tags",
        "is_user_cancellation_event",
    ),
    "utc_now": ("chartreux.core.utils.time", "utc_now"),
    "is_dangerous_directory": ("chartreux.utils.paths", "is_dangerous_directory"),
    "get_platform_display_name": (
        "chartreux.utils.platform",
        "get_platform_display_name",
    ),
    "get_platform_id": ("chartreux.utils.platform", "get_platform_id"),
    "get_platform_version": ("chartreux.utils.platform", "get_platform_version"),
}


def __getattr__(name: str) -> object:
    if name in _MAPPING:
        import importlib

        module_name, attr_name = _MAPPING[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
