"""Scaffolding self-tests for the added §6.3 regression-recording mechanics.

These nodes exercise the shared plumbing in ``tests.differential.regressions``
without any carrier logic: the case registry, the ledger recording path, the
checked-in baseline refresh, and the both-mode gate's derived expectations. The
case nodes themselves land with their carrier owners.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.differential import ledger, regressions

EXPECTED_CASES = {
    "concurrent-recovery-admission": ("C1g", ledger.FAMILY_CONCURRENT_RECOVERY),
    "mcp-partial-convergence": ("C1h", ledger.FAMILY_MCP_CONVERGENCE),
    "mutation-timeout": ("C2c", ledger.FAMILY_MUTATION_TIMEOUT),
    "resume": ("C2b", ledger.FAMILY_RESUME),
    "revocation-race": ("C1c", ledger.FAMILY_REVOCATION_RACE),
    "root-death": ("C2b", ledger.FAMILY_ROOT_DEATH),
    "stale-snapshot": ("C2a", ledger.FAMILY_STALE_SNAPSHOT),
}


def _frames() -> tuple[dict[str, object], ...]:
    return (
        {"type": "request", "method": "session/stop"},
        {"type": "receipt", "status": "settled"},
    )


def _spec(
    *,
    case_id: str = regressions.MUTATION_TIMEOUT,
    owner: str = "C2c",
    family: str = ledger.FAMILY_MUTATION_TIMEOUT,
    trace_module: str = "tests/differential/test_p_parity.py",
    records_to_ledger: bool = False,
    baseline: bool = True,
    landed: bool = False,
) -> regressions.CaseSpec:
    return regressions.CaseSpec(
        case_id=case_id,
        owner=owner,
        family=family,
        trace_module=trace_module,
        records_to_ledger=records_to_ledger,
        baseline=baseline,
        landed=landed,
    )


def test_registry_declares_every_added_case_once() -> None:
    declared = {spec.case_id: spec for spec in regressions.ADDED_CASES}
    assert len(declared) == len(regressions.ADDED_CASES)
    assert set(declared) == set(EXPECTED_CASES)
    for case_id, (owner, family) in EXPECTED_CASES.items():
        spec = regressions.case(case_id)
        assert spec.owner == owner
        assert spec.family == family
        assert spec.trace_module.startswith("tests/differential/")


def test_registry_tracks_the_landed_cases() -> None:
    # C1c landed the revocation-race node first; it records no ledger trace.
    # C2a then landed the stale-snapshot node, C1g the
    # concurrent-recovery-admission node, C1h the mcp-partial-convergence
    # node, C2b the root-death and resume nodes, and C2c the mutation-timeout
    # node (also no ledger trace); the ledger-recording ones each record one
    # trace in tests/differential/test_parity_root.py.
    assert [spec.case_id for spec in regressions.landed_cases()] == [
        regressions.MUTATION_TIMEOUT,
        regressions.ROOT_DEATH,
        regressions.RESUME,
        regressions.STALE_SNAPSHOT,
        regressions.CONCURRENT_RECOVERY,
        regressions.REVOCATION_RACE,
        regressions.MCP_PARTIAL_CONVERGENCE,
    ]
    assert regressions.landed_modules() == ("tests/differential/test_parity_root.py",)
    assert regressions.expected_family_minimums() == {
        ledger.FAMILY_ROOT_DEATH: 1,
        ledger.FAMILY_RESUME: 1,
        ledger.FAMILY_STALE_SNAPSHOT: 1,
        ledger.FAMILY_CONCURRENT_RECOVERY: 1,
        ledger.FAMILY_MCP_CONVERGENCE: 1,
    }
    assert regressions.merge_family_minimums({"base": 3}) == {
        "base": 3,
        ledger.FAMILY_ROOT_DEATH: 1,
        ledger.FAMILY_RESUME: 1,
        ledger.FAMILY_STALE_SNAPSHOT: 1,
        ledger.FAMILY_CONCURRENT_RECOVERY: 1,
        ledger.FAMILY_MCP_CONVERGENCE: 1,
    }


def test_landed_cases_extend_the_gate_expectations(monkeypatch) -> None:
    # A hypothetically landed ledger case (the stale-snapshot case itself is
    # already landed, so the hypothetical uses a different one) extends the
    # gate's family minimums on top of the already-landed cases.
    landed = _spec(
        case_id=regressions.MUTATION_TIMEOUT,
        owner="C2c",
        family=ledger.FAMILY_MUTATION_TIMEOUT,
        trace_module="tests/differential/test_parity_root.py",
        records_to_ledger=True,
        landed=True,
    )
    monkeypatch.setattr(regressions, "ADDED_CASES", (*regressions.ADDED_CASES, landed))

    assert regressions.landed_modules() == ("tests/differential/test_parity_root.py",)
    assert regressions.expected_family_minimums() == {
        ledger.FAMILY_ROOT_DEATH: 1,
        ledger.FAMILY_RESUME: 1,
        ledger.FAMILY_STALE_SNAPSHOT: 1,
        ledger.FAMILY_CONCURRENT_RECOVERY: 1,
        ledger.FAMILY_MCP_CONVERGENCE: 1,
        ledger.FAMILY_MUTATION_TIMEOUT: 1,
    }
    assert regressions.merge_family_minimums({ledger.FAMILY_ROOT: 1}) == {
        ledger.FAMILY_ROOT: 1,
        ledger.FAMILY_ROOT_DEATH: 1,
        ledger.FAMILY_RESUME: 1,
        ledger.FAMILY_STALE_SNAPSHOT: 1,
        ledger.FAMILY_CONCURRENT_RECOVERY: 1,
        ledger.FAMILY_MCP_CONVERGENCE: 1,
        ledger.FAMILY_MUTATION_TIMEOUT: 1,
    }


def test_refresh_marker_is_outside_the_config_namespace(monkeypatch) -> None:
    assert not regressions.REFRESH_ENV.startswith("CHARTREUX_")
    monkeypatch.delenv(regressions.REFRESH_ENV, raising=False)
    assert regressions.refresh_enabled() is False
    monkeypatch.setenv(regressions.REFRESH_ENV, "1")
    assert regressions.refresh_enabled() is True


def test_recording_is_marker_gated(monkeypatch) -> None:
    monkeypatch.delenv(regressions.REFRESH_ENV, raising=False)
    monkeypatch.delenv(ledger.LEDGER_ENV, raising=False)
    assert regressions.recording_enabled() is False
    monkeypatch.setenv(regressions.REFRESH_ENV, "1")
    assert regressions.recording_enabled() is True


def test_unknown_case_is_rejected() -> None:
    with pytest.raises(KeyError):
        regressions.case("no-such-case")


def test_refresh_writes_the_baseline_and_a_later_run_matches(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "baseline.json"
    spec = _spec()
    script = {"ops": ["stop"]}
    monkeypatch.setenv(regressions.REFRESH_ENV, "1")
    first = regressions.record_case(spec, script, _frames(), baseline_path=path)

    entry = json.loads(path.read_text(encoding="utf-8"))[spec.case_id]
    assert entry["digest"] == first
    assert entry["frames"] == len(_frames())
    assert entry["key"] == ledger.trace_key(spec.family, script)

    monkeypatch.delenv(regressions.REFRESH_ENV, raising=False)
    second = regressions.record_case(spec, script, _frames(), baseline_path=path)
    assert second == first


def test_baseline_drift_fails_with_the_refresh_instruction(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps({"mutation-timeout": {"key": "k", "digest": "stale", "frames": 0}}),
        encoding="utf-8",
    )
    monkeypatch.delenv(regressions.REFRESH_ENV, raising=False)
    with pytest.raises(AssertionError, match=regressions.REFRESH_ENV):
        regressions.record_case(_spec(), {"ops": []}, _frames(), baseline_path=path)


def test_a_landed_case_without_a_baseline_requires_a_refresh(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(regressions.REFRESH_ENV, raising=False)
    with pytest.raises(AssertionError, match=regressions.REFRESH_ENV):
        regressions.record_case(
            _spec(landed=True),
            {"ops": []},
            _frames(),
            baseline_path=tmp_path / "baseline.json",
        )


def test_recording_feeds_the_shared_ledger(monkeypatch, tmp_path: Path) -> None:
    directory = tmp_path / "ledger"
    monkeypatch.setenv(ledger.LEDGER_ENV, str(directory))
    monkeypatch.delenv(regressions.REFRESH_ENV, raising=False)
    spec = _spec(
        case_id=regressions.ROOT_DEATH,
        owner="C2b",
        family=ledger.FAMILY_ROOT_DEATH,
        trace_module="tests/differential/test_parity_root.py",
        records_to_ledger=True,
        baseline=False,
    )
    script = {"ops": []}
    observed = regressions.record_case(
        spec, script, _frames(), baseline_path=tmp_path / "baseline.json"
    )

    entries = ledger.read(directory)
    assert list(entries) == [ledger.trace_key(spec.family, script)]
    assert entries[ledger.trace_key(spec.family, script)]["digest"] == observed


def test_a_case_that_does_not_use_the_ledger_writes_nothing(
    monkeypatch, tmp_path: Path
) -> None:
    directory = tmp_path / "ledger"
    monkeypatch.setenv(ledger.LEDGER_ENV, str(directory))
    monkeypatch.delenv(regressions.REFRESH_ENV, raising=False)
    regressions.record_case(
        _spec(records_to_ledger=False, baseline=False),
        {"ops": []},
        _frames(),
        baseline_path=tmp_path / "baseline.json",
    )
    assert ledger.read(directory) == {}
