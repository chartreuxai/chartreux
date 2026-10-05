"""Plain recorded-cost text formatted only from supplied wire snapshots."""

from __future__ import annotations

from chartreux.app_server.models import UsageTotals, UsageWindowSummary


def format_usage_cost(summary: UsageTotals | None) -> str:
    """Distinguish unavailable, unpriced, partial, free, and empty usage.

    Known costs are recorded lower bounds, never estimates from token counts.
    A ready empty window is zero; calls without any priced usage are unknown.
    """
    if (
        summary is None
        or summary.state != "ready"
        or isinstance(summary, UsageWindowSummary)
        and summary.degraded
    ):
        return "—"
    if not summary.has_known_cost and (
        summary.requests > 0 or summary.has_unknown_cost
    ):
        return "Unknown"
    suffix = "+" if summary.has_unknown_cost else ""
    return f"${summary.known_cost_usd:.2f}{suffix}"
