from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    numprocesses = config.getoption("numprocesses", default=None)
    if numprocesses:
        raise pytest.UsageError(
            "performance scenarios cannot run with pytest-xdist; pass -n0"
        )
