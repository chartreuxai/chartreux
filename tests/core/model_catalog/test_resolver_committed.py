"""Resolver behavior for committed identities whose catalog entries disappeared.

These codes drive the resume recovery path: a committed model that no longer
resolves must be reported, never silently re-selected.
"""

from __future__ import annotations

import pytest

from chartreux.core.config import ModelConfig
from chartreux.core.model_catalog.resolver import ModelResolutionError, resolver_for
from chartreux.core.session_types import CommittedModelIdentity
from tests.conftest import build_test_vibe_config


def _resolver_config():
    return build_test_vibe_config(
        models=[ModelConfig(name="model-a", provider="mistral", alias="model-a")]
    )


def _identity(
    *, base_model: str = "model-a", provider: str = "mistral"
) -> CommittedModelIdentity:
    return CommittedModelIdentity(
        base_model=base_model,
        provider=provider,
        wire_name=base_model,
        catalog_revision="test-fixture",
    )


def test_resolve_committed_returns_the_stored_deployment() -> None:
    resolved = resolver_for(_resolver_config()).resolve_committed(_identity())

    assert resolved.base_model == "model-a"
    assert resolved.deployment.provider == "mistral"
    assert resolved.deployment.name == "model-a"


def test_resolve_committed_reports_missing_base_model() -> None:
    with pytest.raises(ModelResolutionError) as exc_info:
        resolver_for(_resolver_config()).resolve_committed(
            _identity(base_model="removed-model")
        )

    assert exc_info.value.code == "committed_model_missing"


def test_resolve_committed_reports_missing_deployment() -> None:
    with pytest.raises(ModelResolutionError) as exc_info:
        resolver_for(_resolver_config()).resolve_committed(_identity(provider="other"))

    assert exc_info.value.code == "committed_deployment_missing"


@pytest.mark.parametrize(
    "policy", ["unknown-model", "@unknown-role", "mistral/model-a", "@"]
)
def test_invalid_model_selection_does_not_fall_back_to_valid_default(
    policy: str,
) -> None:
    resolver = resolver_for(_resolver_config())
    assert resolver.resolve("model-a").base_model == "model-a"
    with pytest.raises(ModelResolutionError):
        resolver.resolve_precedence(task=policy, default="model-a")


@pytest.mark.parametrize(
    "allowed_models", [["unknown-model"], ["other/*"], ["@unknown-role"]]
)
def test_zero_matching_model_policy_fails_closed(allowed_models: list[str]) -> None:
    resolver = resolver_for(_resolver_config())
    with pytest.raises(ModelResolutionError) as exc_info:
        resolver.resolve("model-a", allowed_models=allowed_models)
    assert exc_info.value.code == "allowlist_excluded"


def test_zero_compatible_deployments_does_not_ignore_candidate_policy() -> None:
    with pytest.raises(ModelResolutionError) as exc_info:
        resolver_for(_resolver_config()).resolve(
            "model-a", candidate_filter=lambda _: False
        )
    assert exc_info.value.code == "no_compatible_deployment"
