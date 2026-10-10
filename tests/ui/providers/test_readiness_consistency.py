"""Readiness follows the concrete runtime-selected deployment, not available keys."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from chartreux.core.model_catalog.contracts import CatalogWriteResult
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import (
    CatalogSnapshot,
    CatalogStore,
    load_catalog,
    merge_catalog_overlay,
)
from chartreux.core.model_catalog.resolver import ModelResolver, ResolvedModel
from chartreux.core.model_catalog.schema import ProviderDefinition
from chartreux.ui.providers.management_state import (
    ManagementState,
    usable_canonical_models,
)
from tests.ui.providers.test_management_state import state


def _multi_deployment_state() -> ManagementState:
    draft = state()
    catalog = draft.catalog.model_copy(
        update={
            "providers": {
                **draft.catalog.providers,
                "two": draft.catalog.providers["two"].model_copy(
                    update={"api_key_env_var": "TWO"}
                ),
            }
        }
    )
    draft.snapshot = CatalogSnapshot(catalog, "multi")
    draft.set_role_preset("orchestrator", "a", "medium")
    return draft


def _limited_levels_state(
    first: tuple[str, ...] | None, second: tuple[str, ...] | None
) -> ManagementState:
    """Model "a" with declared thinking limits on its two deployments."""
    draft = _multi_deployment_state()
    deployments = draft.catalog.models["a"].deployments
    limited = draft.catalog.models["a"].model_copy(
        update={
            "deployments": (
                deployments[0].model_copy(update={"supported_thinking_levels": first}),
                deployments[1].model_copy(update={"supported_thinking_levels": second}),
            )
        }
    )
    draft.snapshot = CatalogSnapshot(
        draft.catalog.model_copy(
            update={"models": {**draft.catalog.models, "a": limited}}
        ),
        "limited",
    )
    return draft


def _runtime_selected(
    candidate: CatalogSnapshot,
    role: str,
    *,
    allowed_models: Sequence[str] = (),
    thinking_overrides: Mapping[str, str] | None = None,
) -> ResolvedModel:
    """Resolve one @role exactly as ChartreuxConfigSchema.get_active_model does."""
    definition = candidate.catalog.roles[role]
    return ModelResolver(candidate).resolve(
        f"@{role}",
        allowed_models=allowed_models,
        thinking_override=(thinking_overrides or {}).get(definition.model),
    )


def test_credentials_do_not_change_first_selected_deployment_or_readiness() -> None:
    draft = _multi_deployment_state()
    candidate = draft._candidate_snapshot()
    selected = ModelResolver(candidate).resolve("@orchestrator")
    assert selected.provider.api_key_env_var == "ONE"

    for credentials, expected_ready in (
        ({"TWO": "key"}, False),
        ({"ONE": "key", "TWO": "key"}, True),
    ):
        assert (
            draft.validate_preset(
                "orchestrator", lambda name, values=credentials: values.get(name)
            )
            is None
        ) is expected_ready
        assert (
            ModelResolver(draft._candidate_snapshot())
            .resolve("@orchestrator")
            .provider.api_key_env_var
            == "ONE"
        )


def test_candidate_projection_is_repeatable_and_does_not_mutate_draft() -> None:
    draft = _multi_deployment_state()
    before = (draft.snapshot, dict(draft.connections), dict(draft.role_presets))
    first = draft._candidate_snapshot()
    second = draft._candidate_snapshot()
    assert first.catalog == second.catalog
    assert (draft.snapshot, draft.connections, draft.role_presets) == before


def test_staged_connection_credential_is_used_after_runtime_selection() -> None:
    draft = _multi_deployment_state()
    draft.connections["one"] = replace(draft.connection, api_key_env_var="NEW_ONE")
    validation = draft.validate(
        completion=True,
        credential_resolver=lambda name: "key" if name == "NEW_ONE" else None,
    )
    assert "orchestrator" not in validation.unusable_roles
    assert (
        ModelResolver(draft._candidate_snapshot())
        .resolve("@orchestrator")
        .provider.api_key_env_var
        == "NEW_ONE"
    )


def test_staged_pending_model_participates_in_readiness_selection() -> None:
    draft = _multi_deployment_state()
    draft.for_provider("two")
    draft.select("fresh")
    draft.set_role_preset("cross", "fresh", "low")
    candidate = draft._candidate_snapshot()

    selected = _runtime_selected(candidate, "cross")
    assert (selected.deployment.provider, selected.deployment.name) == ("two", "fresh")
    assert "fresh" in candidate.catalog.models

    resolver = lambda env: "key" if env == "TWO" else None
    assert draft.validate_preset("cross", resolver) is None
    reason = draft.validate_preset("cross", lambda env: None)
    assert reason is not None and "TWO" in reason
    assert "fresh" in usable_canonical_models(candidate, resolver)
    assert "fresh" not in usable_canonical_models(candidate, lambda env: None)

    # Selection restrictions apply to the staged pending deployment as at runtime.
    assert (
        draft.validate_preset("cross", resolver, allowed_models=("two/fresh",)) is None
    )
    assert (
        draft.validate_preset("cross", resolver, allowed_models=("one/*",)) is not None
    )

    # A disabled pending model no longer participates.
    draft.pending["fresh"] = replace(draft.pending["fresh"], enabled=False)
    assert "fresh" not in draft._candidate_snapshot().catalog.models
    assert draft.validate_preset("cross", lambda env: "key") is not None


def test_disabled_deployments_are_excluded_from_readiness_selection() -> None:
    draft = _multi_deployment_state()
    draft.toggle("a", False)
    candidate = draft._candidate_snapshot()

    selected = _runtime_selected(candidate, "orchestrator")
    assert (selected.deployment.provider, selected.deployment.name) == (
        "two",
        "other-a",
    )

    # Readiness gates on the second deployment's credential, not the disabled one's.
    assert (
        draft.validate_preset(
            "orchestrator", lambda env: "key" if env == "TWO" else None
        )
        is None
    )
    reason = draft.validate_preset(
        "orchestrator", lambda env: "key" if env == "ONE" else None
    )
    assert reason is not None and "TWO" in reason

    # Every deployment disabled: the preset is not ready.
    draft.for_provider("two")
    draft.toggle("a", False)
    assert draft.validate_preset("orchestrator", lambda env: "key") is not None


@pytest.mark.parametrize(
    "allowed_models",
    [(), ("a",), ("one/wire-a",), ("two/other-a",), ("two/*",), ("b",), ("absent",)],
)
@pytest.mark.parametrize(
    "credentials", [{}, {"ONE": "key"}, {"TWO": "key"}, {"ONE": "key", "TWO": "key"}]
)
def test_allowed_models_restrictions_match_runtime_resolution(
    allowed_models: tuple[str, ...], credentials: dict[str, str]
) -> None:
    draft = _multi_deployment_state()
    resolver = lambda env: credentials.get(env)
    candidate = draft._candidate_snapshot()

    try:
        selected = _runtime_selected(
            candidate, "orchestrator", allowed_models=allowed_models
        )
    except ValueError:
        selected = None
    expected_ready = selected is not None and (
        not selected.provider.api_key_env_var
        or bool(resolver(selected.provider.api_key_env_var))
    )
    reason = draft.validate_preset(
        "orchestrator", resolver, allowed_models=allowed_models
    )
    assert (reason is None) is expected_ready
    if selected is not None and not expected_ready:
        # Readiness gates on exactly the deployment the runtime selects.
        assert reason is not None
        assert selected.provider.api_key_env_var in reason


@pytest.mark.parametrize(
    ("first_levels", "second_levels", "staged", "override"),
    [
        (("low", "medium"), None, "medium", None),
        (("low", "medium"), None, "medium", "low"),
        (("low", "medium"), None, "medium", "high"),
        (("low", "medium"), None, "high", None),
        (("low", "medium"), None, "high", "low"),
        (("low", "medium"), ("low",), "medium", "high"),
        (None, None, "medium", "max"),
    ],
)
@pytest.mark.parametrize(
    "credentials", [{}, {"ONE": "key"}, {"TWO": "key"}, {"ONE": "key", "TWO": "key"}]
)
def test_thinking_overrides_match_runtime_resolution(
    first_levels: tuple[str, ...] | None,
    second_levels: tuple[str, ...] | None,
    staged: str,
    override: str | None,
    credentials: dict[str, str],
) -> None:
    draft = _limited_levels_state(first_levels, second_levels)
    draft.set_role_preset("orchestrator", "a", staged)
    overrides = {"a": override} if override is not None else None
    resolver = lambda env: credentials.get(env)
    candidate = draft._candidate_snapshot()

    try:
        selected = _runtime_selected(
            candidate, "orchestrator", thinking_overrides=overrides
        )
    except ValueError:
        selected = None
    expected_ready = selected is not None and (
        not selected.provider.api_key_env_var
        or bool(resolver(selected.provider.api_key_env_var))
    )
    reason = draft.validate_preset(
        "orchestrator", resolver, thinking_overrides=overrides
    )
    assert (reason is None) is expected_ready
    if selected is not None and not expected_ready:
        # Readiness gates on exactly the deployment the runtime selects.
        assert reason is not None
        assert selected.provider.api_key_env_var in reason


def test_unsupported_thinking_override_reports_the_effective_level() -> None:
    draft = _limited_levels_state(("low", "medium"), ("low",))
    overrides = {"a": "high"}
    candidate = draft._candidate_snapshot()
    with pytest.raises(ValueError, match="thinking level 'high'"):
        _runtime_selected(candidate, "orchestrator", thinking_overrides=overrides)

    reason = draft.validate_preset(
        "orchestrator", lambda env: "key", thinking_overrides=overrides
    )
    assert reason is not None
    assert "thinking high is unsupported" in reason
    finish = draft.validate(
        credential_resolver=lambda env: "key", thinking_overrides=overrides
    )
    assert "orchestrator" in finish.unusable_roles


def test_usable_canonical_models_honor_allowed_models_restrictions() -> None:
    draft = _multi_deployment_state()
    candidate = draft._candidate_snapshot()
    resolver = lambda env: "key"
    assert usable_canonical_models(candidate, resolver) == frozenset({"a", "b", "c"})
    assert usable_canonical_models(
        candidate, resolver, allowed_models=("one/*",)
    ) == frozenset({"a", "b"})
    assert usable_canonical_models(
        candidate, resolver, allowed_models=("two/*",)
    ) == frozenset({"a", "c"})
    assert usable_canonical_models(
        candidate, resolver, allowed_models=("b",)
    ) == frozenset({"b"})
    # Thinking overrides do not restrict plain-model usability, as at runtime.
    assert usable_canonical_models(
        candidate, resolver, thinking_overrides={"a": "high"}
    ) == usable_canonical_models(candidate, resolver)


def test_persistence_path_readiness_matches_staged_candidate(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    draft = ManagementState.from_snapshot(load_catalog(path), "mistral")
    draft.stage_provider(
        "zai",
        ProviderDefinition(
            api_base="https://api.z.ai/v1", api_key_env_var="ZAI_API_KEY"
        ),
    )
    draft.select("glm-5-3-direct")
    draft.set_role_preset("scout", "glm-5-3-direct", "low")
    candidate = draft._candidate_snapshot()

    resolver = lambda env: "key" if env == "ZAI_API_KEY" else None
    staged_scout = draft.validate_preset("scout", resolver)
    staged_missing = draft.validate_preset("scout", lambda env: None)
    staged_orchestrator = draft.validate_preset("orchestrator", resolver)
    staged_finish = draft.validate(credential_resolver=resolver)
    staged_selected = _runtime_selected(candidate, "scout")
    staged_usable = usable_canonical_models(candidate, resolver)

    result = CatalogStore(path).apply_changes(draft.changes())
    assert isinstance(result, CatalogWriteResult)
    assert result.changed
    saved = load_catalog(path)
    assert saved.catalog == candidate.catalog
    assert saved.revision == candidate.revision

    saved_state = ManagementState.from_snapshot(saved, "zai")
    assert saved_state.validate_preset("scout", resolver) == staged_scout
    assert saved_state.validate_preset("scout", lambda env: None) == staged_missing
    assert saved_state.validate_preset("orchestrator", resolver) == staged_orchestrator
    assert (
        saved_state.validate(credential_resolver=resolver).unusable_roles
        == staged_finish.unusable_roles
    )
    assert usable_canonical_models(saved, resolver) == staged_usable
    saved_selected = _runtime_selected(saved, "scout")
    assert (saved_selected.deployment.provider, saved_selected.deployment.name) == (
        staged_selected.deployment.provider,
        staged_selected.deployment.name,
    )

    # Host restrictions evaluate identically against the staged candidate and the
    # saved catalog.
    allowed = ("zai/glm-5-3-direct",)
    overrides = {"glm-5-3-direct": "low"}
    assert draft.validate_preset(
        "scout", resolver, allowed_models=allowed, thinking_overrides=overrides
    ) == saved_state.validate_preset(
        "scout", resolver, allowed_models=allowed, thinking_overrides=overrides
    )
    assert draft.validate_preset(
        "orchestrator", resolver, allowed_models=allowed, thinking_overrides=overrides
    ) == saved_state.validate_preset(
        "orchestrator", resolver, allowed_models=allowed, thinking_overrides=overrides
    )


def test_glm_style_two_deployment_credential_matrix() -> None:
    """Plan regression: Mistral-first GLM with only a ZAI credential is not ready."""
    overlay = {
        "providers": {
            "zai": {"api_base": "https://api.z.ai/v1", "api_key_env_var": "ZAI_API_KEY"}
        },
        "models": {
            "glm-5-3": {
                "deployments": [
                    {"provider": "mistral"},
                    {"provider": "zai", "name": "glm-5-3"},
                ]
            }
        },
    }
    snapshot = CatalogSnapshot(merge_catalog_overlay(SHIPPED_CATALOG, overlay), "glm")
    draft = ManagementState.from_snapshot(snapshot, "mistral")
    only_zai = lambda env: "key" if env == "ZAI_API_KEY" else None
    both = lambda env: "key"

    # Runtime selection order keeps the shipped Mistral deployment first.
    selected = ModelResolver(snapshot).resolve("@orchestrator")
    assert (selected.deployment.provider, selected.deployment.name) == (
        "mistral",
        "zai-glm-5-3",
    )

    # Only the ZAI credential: the runtime-selected Mistral deployment has no
    # credential, so the preset must not report ready.
    reason = draft.validate_preset("orchestrator", only_zai)
    assert reason is not None and "MISTRAL_API_KEY" in reason
    assert "orchestrator" in draft.validate(credential_resolver=only_zai).unusable_roles
    assert "glm-5-3" not in usable_canonical_models(snapshot, only_zai)

    # Both credentials: ready, and the Mistral deployment stays selected.
    assert draft.validate_preset("orchestrator", both) is None
    assert not draft.validate(credential_resolver=both).unusable_roles
    assert "glm-5-3" in usable_canonical_models(snapshot, both)
    resolved = ModelResolver(draft._candidate_snapshot()).resolve("@orchestrator")
    assert resolved.deployment.provider == "mistral"
