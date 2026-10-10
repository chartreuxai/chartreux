from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.model_catalog import materialization
from chartreux.core.model_catalog.contracts import ProviderWorkbenchResult
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import _snapshot, load_catalog
from chartreux.core.model_catalog.materialization import render_default_template
from chartreux.setup.onboarding import OnboardingFailure, run_onboarding


class _ResultApp:
    def __init__(self, result: OnboardingFailure | ProviderWorkbenchResult) -> None:
        self._result = result

    def run(self) -> OnboardingFailure | ProviderWorkbenchResult:
        return self._result


class _StubConfig:
    def __init__(self, activation_error: Exception | None = None) -> None:
        self._activation_error = activation_error

    def require_active_provider_api_key(self) -> None:
        if self._activation_error is not None:
            raise self._activation_error


class _StubOrchestrator:
    def __init__(self, activation_error: Exception | None = None) -> None:
        self.config = _StubConfig(activation_error)

    async def reload(self) -> None:
        return None


def test_completed_onboarding_materializes_the_default_template(
    config_dir: Path,
) -> None:
    run_onboarding(
        app=_ResultApp(ProviderWorkbenchResult("completed")),  # type: ignore[arg-type]
        orchestrator=_StubOrchestrator(),  # type: ignore[arg-type]
    )

    catalog_path = config_dir / "models.toml"
    assert catalog_path.read_text(encoding="utf-8") == render_default_template()
    # The materialized file serves exactly the in-code defaults.
    snapshot = load_catalog(catalog_path)
    assert snapshot.catalog == SHIPPED_CATALOG
    assert snapshot.overlaid_providers == frozenset()
    assert snapshot.revision == _snapshot(SHIPPED_CATALOG).revision


def test_fsync_failure_warns_without_blocking_completed_onboarding(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    catalog_path = config_dir / "models.toml"
    link_calls = 0
    original_link = materialization.os.link

    def counted_link(source: Path, target: Path) -> None:
        nonlocal link_calls
        link_calls += 1
        original_link(source, target)

    def fail_directory_fsync(directory: Path) -> None:
        raise OSError("directory fsync failed")

    monkeypatch.setattr(materialization.os, "link", counted_link)
    monkeypatch.setattr(materialization, "_fsync_directory", fail_directory_fsync)

    result = run_onboarding(
        app=_ResultApp(ProviderWorkbenchResult("completed")),  # type: ignore[arg-type]
        orchestrator=_StubOrchestrator(),  # type: ignore[arg-type]
    )

    assert result is not None
    assert catalog_path.read_text(encoding="utf-8") == render_default_template()
    assert link_calls == 1
    assert "published, durability uncertain" in caplog.text
    assert "(not published)" not in caplog.text


def test_completed_onboarding_never_overwrites_a_saved_catalog(
    config_dir: Path,
) -> None:
    catalog_path = config_dir / "models.toml"
    saved = b'[providers.custom]\napi_base = "https://custom.test"\n'
    catalog_path.write_bytes(saved)

    run_onboarding(
        app=_ResultApp(ProviderWorkbenchResult("completed", changed=True)),  # type: ignore[arg-type]
        orchestrator=_StubOrchestrator(),  # type: ignore[arg-type]
    )

    assert catalog_path.read_bytes() == saved


def test_cancelled_onboarding_does_not_materialize(config_dir: Path) -> None:
    with pytest.raises(SystemExit) as error:
        run_onboarding(
            app=_ResultApp(ProviderWorkbenchResult("cancelled")),  # type: ignore[arg-type]
            orchestrator=_StubOrchestrator(),  # type: ignore[arg-type]
        )

    assert error.value.code == 0
    assert not (config_dir / "models.toml").exists()


def test_failed_onboarding_does_not_materialize(config_dir: Path) -> None:
    with pytest.raises(SystemExit) as error:
        run_onboarding(
            app=_ResultApp(OnboardingFailure("disk unavailable")),  # type: ignore[arg-type]
            orchestrator=_StubOrchestrator(),  # type: ignore[arg-type]
        )

    assert error.value.code == 1
    assert not (config_dir / "models.toml").exists()


def test_onboarding_without_an_activated_provider_does_not_materialize(
    config_dir: Path,
) -> None:
    with pytest.raises(SystemExit) as error:
        run_onboarding(
            app=_ResultApp(ProviderWorkbenchResult("completed")),  # type: ignore[arg-type]
            orchestrator=_StubOrchestrator(activation_error=ValueError("no key")),  # type: ignore[arg-type]
        )

    assert error.value.code == 1
    assert not (config_dir / "models.toml").exists()
