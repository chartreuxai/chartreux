from __future__ import annotations

import sys

import pytest

from tests.perf._metrics import machine_context, percentiles, record

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="performance scenarios require Linux /proc readers"
)


@pytest.mark.perf
@pytest.mark.timeout(60)
def test_perf_metrics_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    record("smoke", {"ok": True})
    assert capsys.readouterr().out.strip() == 'PERF-BLOB:smoke:{"ok":true}'

    summary = percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary == {
        "min": 1.0,
        "p50": 3.0,
        "p95": 4.8,
        "p99": 4.96,
        "max": 5.0,
        "mean": 3.0,
    }

    context = machine_context()
    assert isinstance(context, dict)
    assert {
        "cpu_model",
        "core_count",
        "python_version",
        "platform",
        "loadavg",
        "commit",
    } <= context.keys()
