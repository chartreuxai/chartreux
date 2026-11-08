from __future__ import annotations

import pytest

from tests.perf._runtime_metrics import PerfPathError, resolve_perf_paths


def pytest_configure(config: pytest.Config) -> None:
    numprocesses = config.getoption("numprocesses", default=None)
    if numprocesses:
        raise pytest.UsageError(
            "performance scenarios cannot run with pytest-xdist; pass -n0"
        )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Fail closed when the measurement paths escape the worker boundary."""
    try:
        resolve_perf_paths()
    except PerfPathError as error:
        raise pytest.UsageError(str(error)) from error
