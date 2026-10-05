from __future__ import annotations


def process_name() -> str:
    # Shown in Activity Monitor / ps / top so Chartreux can be spotted and killed.
    # Concurrent instances are differentiated by the process manager's own PID
    # column, so the name itself stays clean.
    return "Chartreux CLI"
