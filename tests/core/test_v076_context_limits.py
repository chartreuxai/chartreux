"""v0.7.6 P4 — compaction threshold representation and provider context limits.

Slice ownership (package P4's chunks append their own slices below this header):

* Slice A (chunk 4a) — the provider-fact fields (context limit and output
  reserve source), their shipped-catalog values, the materialize carrier, and
  the projection switch that stops reading ``auto_compact_threshold`` as the
  provider's context window. Slice A landed NO threshold raise:
  ``TestPreRaiseShippedThresholds`` pinned the literal pre-raise values, and
  chunk 4c OWNS rewriting that class into the raise assertion.
* Slice B (chunk 4b) — the load/materialize invariant and its fail-closed
  UNKNOWN-limit semantics.
* Slice C (chunk 4c) — the conditional raises and the child default. The 4a
  guard is rewritten as ``TestRaisedShippedThresholds``: the raise is present
  only where the deployment documents both facts and the invariant holds at the
  raised value, and the child default is gated by the same three terms.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from pydantic import ValidationError
import pytest

from chartreux.app_server._resources import ResourceRequestHandler
from chartreux.app_server._runtime import build_unified_runtime_snapshot
from chartreux.core.agents.identity_launch import resolve_identity_launch
from chartreux.core.agents.launch import resolve_launch
from chartreux.core.config import DEFAULT_AUTO_COMPACT_THRESHOLD, ChartreuxConfigSchema
from chartreux.core.config.chartreux_schema import (
    ONE_TURN_GROWTH_ALLOWANCE,
    compaction_threshold_violations,
)
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.models import ModelConfig
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.middleware import (
    ContextWarningMiddleware,
    ConversationContext,
    MiddlewareAction,
)
from chartreux.core.model_catalog import defaults as defaults_module
from chartreux.core.model_catalog.defaults import (
    CHILD_AUTO_COMPACT_THRESHOLD,
    SHIPPED_CATALOG,
    child_auto_compact_threshold,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot, merge_catalog_overlay
from chartreux.core.model_catalog.resolver import (
    ModelResolutionError,
    ModelResolver,
    ResolvedModel,
    resolver_for,
)
from chartreux.core.model_catalog.schema import (
    BaseModelDefinition,
    DeploymentDefinition,
    ModelCatalog,
    ProviderDefinition,
)
from chartreux.core.session_types import AgentStats, CommittedModelIdentity
from chartreux.utils import VIBE_WARNING_TAG
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

#: Provider-documented facts per shipped deployment: (context limit, output
#: reserve). ``None`` is an undocumented fact, never a guessed number.
SHIPPED_PROVIDER_FACTS: dict[tuple[str, str, str], tuple[int | None, int | None]] = {
    ("glm-5-3", "mistral", "zai-glm-5-3"): (1_000_000, 128_000),
    ("mistral-large-4", "mistral", "mistral-large-4"): (1_000_000, None),
}

#: Deployments shipping no output reserve: the provider documents no default
#: output budget, so no reserve can be assumed and no raise can be justified.
NO_DOCUMENTED_RESERVE: frozenset[tuple[str, str, str]] = frozenset({
    ("mistral-large-4", "mistral", "mistral-large-4")
})

#: The decided shipped thresholds after the 4c raise: 500k only where the
#: deployment documents BOTH provider facts and the invariant holds at the
#: raised value; every other deployment keeps its prior 400k.
RAISED_SHIPPED_THRESHOLDS: dict[tuple[str, str, str], int] = {
    ("glm-5-3", "mistral", "zai-glm-5-3"): 500_000,
    ("mistral-large-4", "mistral", "mistral-large-4"): 400_000,
}

#: The shipped raise's target and the prior threshold it raises from.
SHIPPED_RAISE = 500_000
PRIOR_SHIPPED_THRESHOLD = 400_000

#: The shipped deployments that KEEP their prior threshold: they fail the gate
#: on the output-reserve term, so no arithmetic runs on an unknown reserve.
KEEPS_PRIOR_THRESHOLD: frozenset[tuple[str, str, str]] = NO_DOCUMENTED_RESERVE

#: 800k is a per-deployment value only: it never ships, and the invariant
#: rejects it wherever the budget cannot hold it.
RAISE_BEYOND_THE_BUDGET = 800_000

#: A raise above the shipped 500k baseline that still leaves room on
#: zai-glm-5-3: 600k + 217k one-turn allowance stays inside 1M - 128k.
RAISE_WITH_ROOM = 600_000


class _ActiveModelConfig:
    """The narrow config surface the context-warning middleware reads."""

    def __init__(self, model: ModelConfig) -> None:
        self._model = model

    def get_active_model(self) -> ModelConfig:
        return self._model


def _shipped_deployments() -> dict[tuple[str, str, str], DeploymentDefinition]:
    return {
        (base, deployment.provider, deployment.name): deployment
        for base, definition in SHIPPED_CATALOG.models.items()
        for deployment in definition.deployments
    }


def _model(*, limit: int | None, threshold: int = 400) -> ModelConfig:
    return ModelConfig(
        name="wire-model",
        provider="mistral",
        alias="probe",
        auto_compact_threshold=threshold,
        context_limit=limit,
    )


def _config_with(model: ModelConfig) -> ChartreuxConfigSchema:
    return build_test_vibe_config(models=[model], active_model=model.alias)


def _undocumented_config() -> ChartreuxConfigSchema:
    return _config_with(
        ModelConfig(name="undocumented-wire", provider="mistral", alias="undocumented")
    )


def _request_handler_context_window(config: ChartreuxConfigSchema) -> int | None:
    handler = ResourceRequestHandler.__new__(ResourceRequestHandler)
    handler._agent_loop = SimpleNamespace(config=config)  # type: ignore[assignment]
    return handler._context_window()


def _runtime_snapshot_context_window(config: ChartreuxConfigSchema) -> int | None:
    orchestrator = SimpleNamespace(config=config, layers=())
    snapshot = build_unified_runtime_snapshot(cast(Any, orchestrator), cast(Any, None))
    return snapshot.context_window


def _context(model: ModelConfig, *, context_tokens: int) -> ConversationContext:
    return ConversationContext(
        messages=cast(Any, None),
        stats=AgentStats(context_tokens=context_tokens),
        config=cast(Any, _ActiveModelConfig(model)),
    )


# --- Slice A -------------------------------------------------------------


class TestDeploymentLimitFacts:
    """The two provider facts parse as whole token counts or as UNKNOWN."""

    def test_a_documented_limit_and_reserve_parse_as_whole_token_counts(self) -> None:
        deployment = DeploymentDefinition.model_validate({
            "provider": "mistral",
            "name": "wire-model",
            "context_limit": 1_048_576,
            "max_completion_tokens": 131_072.0,
        })

        assert deployment.context_limit == 1_048_576
        assert deployment.max_completion_tokens == 131_072

    def test_an_undocumented_fact_stays_unknown(self) -> None:
        deployment = DeploymentDefinition(provider="mistral", name="wire-model")

        assert deployment.context_limit is None
        assert deployment.max_completion_tokens is None

        explicit = DeploymentDefinition(
            provider="mistral",
            name="wire-model",
            context_limit=None,
            max_completion_tokens=None,
        )

        assert explicit.context_limit is None
        assert explicit.max_completion_tokens is None

    @pytest.mark.parametrize("field", ["context_limit", "max_completion_tokens"])
    @pytest.mark.parametrize("rejected", [0, -1, 1.5, True, "1000"])
    def test_a_non_token_count_is_rejected(self, field: str, rejected: Any) -> None:
        with pytest.raises(ValidationError):
            DeploymentDefinition.model_validate({
                "provider": "mistral",
                "name": "wire-model",
                field: rejected,
            })


class TestShippedLimitAndReserveProvenance:
    """The shipped catalog carries the documented facts and nothing more."""

    def test_every_shipped_deployment_matches_its_documented_facts(self) -> None:
        actual = {
            key: (deployment.context_limit, deployment.max_completion_tokens)
            for key, deployment in _shipped_deployments().items()
        }

        assert actual == SHIPPED_PROVIDER_FACTS

    def test_the_no_documented_reserve_set_is_exactly_the_undocumented_ones(
        self,
    ) -> None:
        actual = frozenset(
            key
            for key, deployment in _shipped_deployments().items()
            if deployment.max_completion_tokens is None
        )

        assert actual == NO_DOCUMENTED_RESERVE


class TestMaterializeCarriesProviderFacts:
    """The provider facts survive the materialize boundary onto ModelConfig."""

    def test_the_active_model_carries_its_provider_facts(self) -> None:
        active = build_test_vibe_config().get_active_model()

        assert (active.alias, active.name) == ("glm-5-3", "zai-glm-5-3")
        assert active.context_limit == 1_000_000
        assert active.max_completion_tokens == 128_000

    def test_a_deployment_without_a_documented_reserve_materializes_none(self) -> None:
        config = build_test_vibe_config()
        materialized = (
            resolver_for(config)
            .resolve("mistral-large-4")
            .materialize(auto_compact_threshold=config.auto_compact_threshold)
        )

        assert materialized.context_limit == 1_000_000
        assert materialized.max_completion_tokens is None

    def test_an_unknown_limit_stays_unknown_through_materialize(self) -> None:
        deployment = DeploymentDefinition(provider="probe", name="undocumented-wire")
        definition = BaseModelDefinition(thinking="medium", deployments=(deployment,))
        resolved = ResolvedModel(
            "undocumented",
            definition,
            deployment,
            ProviderDefinition(api_base="https://provider.test/v1"),
            "test-revision",
        )

        materialized = resolved.materialize(auto_compact_threshold=200_000)

        assert materialized.context_limit is None
        assert materialized.max_completion_tokens is None
        assert materialized.auto_compact_threshold == 200_000


class TestProjectionCarriesTheLimit:
    """Every context-window projection reports the provider limit, not the threshold."""

    @pytest.mark.asyncio
    async def test_the_warning_states_the_percentage_of_the_provider_limit(
        self,
    ) -> None:
        context = _context(_model(limit=1_000, threshold=400), context_tokens=500)

        result = await ContextWarningMiddleware(0.5).before_turn(context)

        assert result.action is MiddlewareAction.INJECT_MESSAGE
        assert result.message == (
            f"<{VIBE_WARNING_TAG}>You have used 50% of your total context "
            f"(500/1,000 tokens)</{VIBE_WARNING_TAG}>"
        )

    @pytest.mark.asyncio
    async def test_the_compaction_threshold_is_not_a_denominator(self) -> None:
        context = _context(_model(limit=1_000, threshold=400), context_tokens=450)

        result = await ContextWarningMiddleware(0.5).before_turn(context)

        assert result.action is MiddlewareAction.CONTINUE

    @pytest.mark.asyncio
    async def test_an_undocumented_limit_suppresses_the_percentage_warning(
        self,
    ) -> None:
        context = _context(_model(limit=None, threshold=400), context_tokens=10_000_000)

        result = await ContextWarningMiddleware(0.5).before_turn(context)

        assert result.action is MiddlewareAction.CONTINUE

    def test_the_session_stats_projection_uses_the_limit(self) -> None:
        assert _request_handler_context_window(build_test_vibe_config()) == 1_000_000
        assert _request_handler_context_window(_undocumented_config()) is None

    def test_the_runtime_snapshot_projection_uses_the_limit(self) -> None:
        assert _runtime_snapshot_context_window(build_test_vibe_config()) == 1_000_000
        assert _runtime_snapshot_context_window(_undocumented_config()) is None


class TestRaisedShippedThresholds:
    """The 4c raise replaces slice A's pre-raise guard.

    ``RAISED_SHIPPED_THRESHOLDS`` holds the decided values and
    ``KEEPS_PRIOR_THRESHOLD`` the deployments that fail the gate. The raise is
    never assumed from a known value: each deployment's threshold is checked
    against the invariant computed independently here.
    """

    def test_every_shipped_deployment_carries_its_decided_threshold(self) -> None:
        actual = {
            key: deployment.auto_compact_threshold
            for key, deployment in _shipped_deployments().items()
        }

        assert actual == RAISED_SHIPPED_THRESHOLDS

    def test_the_raise_is_present_exactly_where_the_invariant_holds(self) -> None:
        for key, deployment in _shipped_deployments().items():
            limit = deployment.context_limit
            reserve = deployment.max_completion_tokens
            holds = (
                limit is not None
                and reserve is not None
                and SHIPPED_RAISE + ONE_TURN_GROWTH_ALLOWANCE <= limit - reserve
            )

            assert deployment.auto_compact_threshold == (
                SHIPPED_RAISE if holds else PRIOR_SHIPPED_THRESHOLD
            ), key

    def test_the_named_keep_prior_deployments_fail_the_gate(self) -> None:
        kept = frozenset(
            key
            for key, deployment in _shipped_deployments().items()
            if deployment.auto_compact_threshold != SHIPPED_RAISE
        )

        assert kept == KEEPS_PRIOR_THRESHOLD
        for key in kept:
            deployment = _shipped_deployments()[key]
            assert deployment.context_limit is not None, key
            assert deployment.max_completion_tokens is None, key
            assert (
                SHIPPED_RAISE + ONE_TURN_GROWTH_ALLOWANCE
                > deployment.context_limit - PRIOR_SHIPPED_THRESHOLD
            ), key

    def test_every_raised_deployment_ships_a_known_limit_and_reserve(self) -> None:
        raised = [
            (key, deployment)
            for key, deployment in _shipped_deployments().items()
            if deployment.auto_compact_threshold is not None
            and deployment.auto_compact_threshold > PRIOR_SHIPPED_THRESHOLD
        ]

        assert raised
        for key, deployment in raised:
            assert deployment.auto_compact_threshold == SHIPPED_RAISE, key
            assert deployment.context_limit is not None, key
            assert deployment.max_completion_tokens is not None, key

    def test_the_migration_sentence_ships_with_the_raise(self) -> None:
        documented = " ".join((defaults_module.__doc__ or "").split())

        assert (
            "every shipped deployment receiving the 500k raise ships with a "
            "KNOWN context limit AND a KNOWN output reserve"
        ) in documented
        assert (
            "must populate the limit field, or the overlay's file fails "
            "validation at load"
        ) in documented
        assert "nothing silently raises, nothing silently stays" in documented

    def test_the_materialized_threshold_is_the_raised_value(self) -> None:
        config = build_test_vibe_config()

        assert config.get_active_model().auto_compact_threshold == SHIPPED_RAISE
        assert config.auto_compact_threshold == DEFAULT_AUTO_COMPACT_THRESHOLD

    def test_eight_hundred_thousand_ships_nowhere_and_is_gated(self) -> None:
        assert SHIPPED_RAISE < RAISE_BEYOND_THE_BUDGET
        assert all(
            deployment.auto_compact_threshold != RAISE_BEYOND_THE_BUDGET
            for deployment in _shipped_deployments().values()
        )
        glm = _shipped_deployments()[("glm-5-3", "mistral", "zai-glm-5-3")]
        assert glm.context_limit is not None
        assert glm.max_completion_tokens is not None
        assert (
            RAISE_BEYOND_THE_BUDGET + ONE_TURN_GROWTH_ALLOWANCE
            > glm.context_limit - glm.max_completion_tokens
        )
        with pytest.raises(ValidationError, match="one-turn growth allowance"):
            _load(_raised("glm-5-3", RAISE_BEYOND_THE_BUDGET))

    def test_neither_provider_fact_feeds_the_threshold(self) -> None:
        active = build_test_vibe_config().get_active_model()

        assert active.context_limit is not None
        assert active.max_completion_tokens is not None
        assert active.auto_compact_threshold not in {
            active.context_limit,
            active.context_limit - active.max_completion_tokens,
        }


# --- Slice B -------------------------------------------------------------


def _deployments_snapshot(patches: dict[str, dict[str, Any]]) -> CatalogSnapshot:
    """The shipped catalog with the named deployments' fields replaced."""
    models: dict[str, Any] = {}
    for base_model, patch in patches.items():
        deployment = SHIPPED_CATALOG.models[base_model].deployments[0]
        models[base_model] = {
            "deployments": [{"provider": deployment.provider, **patch}]
        }
    return CatalogSnapshot(
        merge_catalog_overlay(SHIPPED_CATALOG, {"models": models}), "raised-catalog"
    )


def _raised(base_model: str, threshold: int | None) -> CatalogSnapshot:
    return _deployments_snapshot({base_model: {"auto_compact_threshold": threshold}})


def _load(
    snapshot: CatalogSnapshot | None = None, **config: Any
) -> ChartreuxConfigSchema:
    """Validate one configuration the way config load does, catalog included."""
    context = None if snapshot is None else {"catalog_snapshot": snapshot}
    return ChartreuxConfigSchema.model_validate(dict(config), context=context)


class TestThresholdInvariantAtLoad:
    """Config load rejects a catalog whose raises leave no measured headroom."""

    def test_the_shipped_catalog_sits_inside_the_invariant(self) -> None:
        assert (
            compaction_threshold_violations(
                SHIPPED_CATALOG, fallback_threshold=DEFAULT_AUTO_COMPACT_THRESHOLD
            )
            == []
        )

    def test_the_shipped_catalog_loads_unchanged(self) -> None:
        config = _load(CatalogSnapshot(SHIPPED_CATALOG, "shipped"))

        assert config.auto_compact_threshold == DEFAULT_AUTO_COMPACT_THRESHOLD

    def test_a_raise_that_leaves_room_for_one_turn_loads(self) -> None:
        _load(_raised("glm-5-3", RAISE_WITH_ROOM))

    def test_a_raise_past_the_output_budget_rejects_the_configuration(self) -> None:
        with pytest.raises(ValidationError, match="one-turn growth allowance"):
            _load(_raised("glm-5-3", 900_000))

    def test_an_unknown_limit_raise_rejects_the_configuration(self) -> None:
        raised_without_a_limit = _deployments_snapshot({
            "glm-5-3": {
                "auto_compact_threshold": RAISE_WITH_ROOM,
                "context_limit": None,
            }
        })

        with pytest.raises(ValidationError, match="context limit is unknown"):
            _load(raised_without_a_limit)

    def test_an_unknown_reserve_raise_fails_closed_without_arithmetic(self) -> None:
        assert 500_000 + ONE_TURN_GROWTH_ALLOWANCE < 1_000_000

        with pytest.raises(
            ValidationError, match="no provider output reserve is documented"
        ):
            _load(_raised("mistral-large-4", 500_000))

    def test_a_raised_fallback_on_an_unthresholded_deployment_is_rejected(self) -> None:
        unthresholded = _deployments_snapshot({
            "glm-5-3": {"auto_compact_threshold": None, "context_limit": None}
        })

        with pytest.raises(ValidationError, match="context limit is unknown"):
            _load(unthresholded, auto_compact_threshold=RAISE_WITH_ROOM)

    def test_the_error_names_every_rejected_deployment_and_both_resolution_paths(
        self,
    ) -> None:
        snapshot = _deployments_snapshot({
            "glm-5-3": {
                "auto_compact_threshold": RAISE_WITH_ROOM,
                "context_limit": None,
            },
            "mistral-large-4": {"auto_compact_threshold": 500_000},
        })

        with pytest.raises(ValidationError) as raised:
            _load(snapshot)

        message = str(raised.value)
        assert "mistral/zai-glm-5-3" in message
        assert "mistral/mistral-large-4" in message
        assert "context_limit" in message
        assert "max_completion_tokens" in message
        assert "revert the threshold raise" in message

    def test_one_rejected_deployment_rejects_the_whole_file(self) -> None:
        snapshot = _deployments_snapshot({
            "glm-5-3": {"auto_compact_threshold": 500_000},
            "mistral-large-4": {"auto_compact_threshold": 500_000},
        })

        with pytest.raises(ValidationError) as raised:
            _load(snapshot)

        message = str(raised.value)
        assert "mistral/mistral-large-4" in message
        assert "mistral/zai-glm-5-3" not in message


class TestLastAcceptedConfiguration:
    """A rejected load leaves the last accepted configuration in force."""

    @pytest.mark.asyncio
    async def test_a_first_load_with_a_rejected_catalog_fails_closed(self) -> None:
        session = OverridesLayer(data={"theme": "accepted"})

        with pytest.raises(ValueError, match="context limit is unknown"):
            await ConfigOrchestrator.create(
                schema=ChartreuxConfigSchema,
                layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), session],
                default_layer_resolver=lambda: session,
                catalog_snapshot=_deployments_snapshot({
                    "glm-5-3": {
                        "auto_compact_threshold": RAISE_WITH_ROOM,
                        "context_limit": None,
                    }
                }),
            )

    @pytest.mark.asyncio
    async def test_a_rejected_reload_keeps_the_last_accepted_configuration(
        self,
    ) -> None:
        session = OverridesLayer(data={"theme": "accepted"})
        orchestrator = await ConfigOrchestrator.create(
            schema=ChartreuxConfigSchema,
            layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), session],
            default_layer_resolver=lambda: session,
            catalog_loader=lambda: _deployments_snapshot({
                "glm-5-3": {
                    "auto_compact_threshold": RAISE_WITH_ROOM,
                    "context_limit": None,
                }
            }),
        )
        before = orchestrator.config, orchestrator.accepted_token

        with pytest.raises(ValueError, match="context limit is unknown"):
            await orchestrator.reload()

        assert (orchestrator.config, orchestrator.accepted_token) == before
        assert (
            orchestrator.config.get_active_model().auto_compact_threshold
            == SHIPPED_RAISE
        )

    @pytest.mark.asyncio
    async def test_an_accepted_reload_adopts_the_validated_catalog(self) -> None:
        session = OverridesLayer(data={"theme": "accepted"})
        orchestrator = await ConfigOrchestrator.create(
            schema=ChartreuxConfigSchema,
            layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), session],
            default_layer_resolver=lambda: session,
            catalog_loader=lambda: _raised("glm-5-3", 500_000),
        )
        before = orchestrator.accepted_token

        await orchestrator.reload()

        assert orchestrator.accepted_token is not before
        assert (
            orchestrator.config.catalog_snapshot.catalog
            .models["glm-5-3"]
            .deployments[0]
            .auto_compact_threshold
            == 500_000
        )


class TestMaterializeRejects:
    """The same invariant is re-validated at every materialize call."""

    def test_a_shipped_raise_past_the_budget_is_rejected_at_materialize(self) -> None:
        resolver = ModelResolver(_raised("glm-5-3", 900_000))

        with pytest.raises(ModelResolutionError) as raised:
            resolver.resolve("glm-5-3").materialize(
                auto_compact_threshold=DEFAULT_AUTO_COMPACT_THRESHOLD
            )

        assert raised.value.code == "compaction_threshold_invalid"
        assert "one-turn growth allowance" in str(raised.value)

    def test_an_unknown_reserve_raise_is_rejected_at_materialize(self) -> None:
        resolver = ModelResolver(_raised("mistral-large-4", 500_000))

        with pytest.raises(
            ModelResolutionError, match="no provider output reserve is documented"
        ):
            resolver.resolve("mistral-large-4").materialize(
                auto_compact_threshold=DEFAULT_AUTO_COMPACT_THRESHOLD
            )

    def test_the_fallback_raise_is_rejected_at_materialize(self) -> None:
        resolver = ModelResolver(
            _deployments_snapshot({
                "glm-5-3": {"auto_compact_threshold": None, "context_limit": None}
            })
        )

        with pytest.raises(ModelResolutionError, match="context limit is unknown"):
            resolver.resolve("glm-5-3").materialize(
                auto_compact_threshold=RAISE_WITH_ROOM
            )

    def test_a_documented_raise_materializes_with_the_raised_value(self) -> None:
        resolver = ModelResolver(_raised("glm-5-3", 500_000))

        materialized = resolver.resolve("glm-5-3").materialize(
            auto_compact_threshold=DEFAULT_AUTO_COMPACT_THRESHOLD
        )

        assert materialized.auto_compact_threshold == 500_000

    def test_every_shipped_deployment_still_materializes(self) -> None:
        resolver = ModelResolver(CatalogSnapshot(SHIPPED_CATALOG, "shipped"))

        for base_model, definition in SHIPPED_CATALOG.models.items():
            materialized = resolver.resolve(base_model).materialize(
                auto_compact_threshold=DEFAULT_AUTO_COMPACT_THRESHOLD
            )

            assert (
                materialized.auto_compact_threshold
                == definition.deployments[0].auto_compact_threshold
            )


# --- Slice C -------------------------------------------------------------


def _probe_catalog(
    *, limit: int | None, reserve: int | None, threshold: int | None = None
) -> ModelCatalog:
    """A one-deployment catalog whose provider facts the test controls."""
    deployment: dict[str, Any] = {"provider": "probe", "name": "probe-wire"}
    if limit is not None:
        deployment["context_limit"] = limit
    if reserve is not None:
        deployment["max_completion_tokens"] = reserve
    if threshold is not None:
        deployment["auto_compact_threshold"] = threshold
    return ModelCatalog.model_validate({
        "providers": {"probe": {"api_base": "https://probe.test/v1"}},
        "models": {"probe-model": {"thinking": "off", "deployments": [deployment]}},
        "roles": {
            "main": {
                "model": "probe-model",
                "thinking": "off",
                "description": "main preset",
            }
        },
    })


def _probe_config(
    *, limit: int | None, reserve: int | None, threshold: int | None = None
) -> ChartreuxConfigSchema:
    base = build_test_vibe_config(active_model="@main")
    return base.attach_catalog_snapshot(
        CatalogSnapshot(
            _probe_catalog(limit=limit, reserve=reserve, threshold=threshold),
            "probe-catalog",
        )
    )


def _deployment(*, limit: int | None, reserve: int | None) -> DeploymentDefinition:
    return DeploymentDefinition(
        provider="mistral",
        name="wire-model",
        context_limit=limit,
        max_completion_tokens=reserve,
    )


def _committed_probe() -> CommittedModelIdentity:
    return CommittedModelIdentity(
        base_model="probe-model",
        provider="probe",
        wire_name="probe-wire",
        catalog_revision="probe-catalog",
        thinking="off",
    )


@pytest.fixture
def materialized_thresholds(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Every threshold a materialize call receives, in call order."""
    seen: list[int] = []
    original = ResolvedModel.materialize

    def recorder(
        self: ResolvedModel, *, auto_compact_threshold: int, **kwargs: Any
    ) -> ModelConfig:
        seen.append(auto_compact_threshold)
        return original(self, auto_compact_threshold=auto_compact_threshold, **kwargs)

    monkeypatch.setattr(ResolvedModel, "materialize", recorder)
    return seen


def _resolve_child(
    config: ChartreuxConfigSchema,
    *,
    committed_model: CommittedModelIdentity | None = None,
):
    return resolve_launch(
        identity="coder",
        config=None,
        parent_orchestrator=FakeConfigOrchestrator(config),
        tool_inventory={"bash": object(), "read_file": object()},
        authorized_tool_names=frozenset({"bash", "read_file"}),
        committed_model=committed_model,
    )


class TestChildDefaultGate:
    """The child default 500k is gated by the same three terms as the raise."""

    @pytest.mark.parametrize(
        ("limit", "reserve", "fallback", "expected"),
        [
            (
                1_000_000,
                128_000,
                DEFAULT_AUTO_COMPACT_THRESHOLD,
                CHILD_AUTO_COMPACT_THRESHOLD,
            ),
            # Exactly one allowance of room left inside limit - reserve.
            (
                845_000,
                128_000,
                DEFAULT_AUTO_COMPACT_THRESHOLD,
                CHILD_AUTO_COMPACT_THRESHOLD,
            ),
            # One token less of budget and the raise no longer fits.
            (
                844_999,
                128_000,
                DEFAULT_AUTO_COMPACT_THRESHOLD,
                DEFAULT_AUTO_COMPACT_THRESHOLD,
            ),
            (
                1_000_000,
                None,
                DEFAULT_AUTO_COMPACT_THRESHOLD,
                DEFAULT_AUTO_COMPACT_THRESHOLD,
            ),
            (
                None,
                128_000,
                DEFAULT_AUTO_COMPACT_THRESHOLD,
                DEFAULT_AUTO_COMPACT_THRESHOLD,
            ),
            (
                None,
                None,
                DEFAULT_AUTO_COMPACT_THRESHOLD,
                DEFAULT_AUTO_COMPACT_THRESHOLD,
            ),
            (
                600_000,
                128_000,
                DEFAULT_AUTO_COMPACT_THRESHOLD,
                DEFAULT_AUTO_COMPACT_THRESHOLD,
            ),
            # Never lowers a configured fallback that already exceeds it.
            (1_000_000, 128_000, RAISE_WITH_ROOM, RAISE_WITH_ROOM),
        ],
    )
    def test_the_gate_decides_the_child_default(
        self, limit: int | None, reserve: int | None, fallback: int, expected: int
    ) -> None:
        deployment = _deployment(limit=limit, reserve=reserve)

        assert child_auto_compact_threshold(deployment, fallback) == expected

    def test_the_boundary_is_the_measured_allowance(self) -> None:
        assert (
            CHILD_AUTO_COMPACT_THRESHOLD + ONE_TURN_GROWTH_ALLOWANCE
            == 845_000 - 128_000
        )

    def test_a_shipped_raised_deployment_takes_the_child_default(self) -> None:
        glm = _shipped_deployments()[("glm-5-3", "mistral", "zai-glm-5-3")]

        assert (
            child_auto_compact_threshold(glm, DEFAULT_AUTO_COMPACT_THRESHOLD)
            == CHILD_AUTO_COMPACT_THRESHOLD
        )

    def test_a_named_keep_prior_deployment_keeps_the_prior_child_default(self) -> None:
        for key in KEEPS_PRIOR_THRESHOLD:
            deployment = _shipped_deployments()[key]

            assert (
                child_auto_compact_threshold(deployment, DEFAULT_AUTO_COMPACT_THRESHOLD)
                == DEFAULT_AUTO_COMPACT_THRESHOLD
            ), key


class TestChildDefaultAtEveryMaterializeSite:
    """All three child materialize sites resolve and validate under the gate."""

    def test_the_resolved_model_site_applies_the_child_default(
        self, materialized_thresholds: list[int]
    ) -> None:
        config = _probe_config(limit=1_000_000, reserve=128_000)

        launch = _resolve_child(config, committed_model=_committed_probe())

        assert materialized_thresholds == [CHILD_AUTO_COMPACT_THRESHOLD]
        assert (
            launch.effective_model.auto_compact_threshold
            == CHILD_AUTO_COMPACT_THRESHOLD
        )

    def test_the_assignment_candidate_site_applies_the_child_default(
        self, materialized_thresholds: list[int]
    ) -> None:
        config = _probe_config(limit=1_000_000, reserve=128_000)

        launch = _resolve_child(config)

        # Candidate assignment first, then the selected model's materialization.
        assert materialized_thresholds == [
            CHILD_AUTO_COMPACT_THRESHOLD,
            CHILD_AUTO_COMPACT_THRESHOLD,
        ]
        assert (
            launch.effective_model.auto_compact_threshold
            == CHILD_AUTO_COMPACT_THRESHOLD
        )

    def test_the_identity_launch_site_applies_the_child_default(
        self, materialized_thresholds: list[int]
    ) -> None:
        config = _probe_config(limit=1_000_000, reserve=128_000)

        resolve_identity_launch(
            {"task": "do work"},
            config=config,
            authorized_tool_names=frozenset({"bash", "read_file"}),
        )

        assert materialized_thresholds == [CHILD_AUTO_COMPACT_THRESHOLD]

    def test_a_deployment_failing_the_gate_keeps_the_prior_child_default(
        self, materialized_thresholds: list[int]
    ) -> None:
        config = _probe_config(limit=None, reserve=128_000)

        launch = _resolve_child(config, committed_model=_committed_probe())
        assert materialized_thresholds == [DEFAULT_AUTO_COMPACT_THRESHOLD]
        assert (
            launch.effective_model.auto_compact_threshold
            == DEFAULT_AUTO_COMPACT_THRESHOLD
        )

        materialized_thresholds.clear()
        no_reserve = _probe_config(limit=1_000_000, reserve=None)
        kept = _resolve_child(no_reserve, committed_model=_committed_probe())
        assert materialized_thresholds == [DEFAULT_AUTO_COMPACT_THRESHOLD]
        assert kept.effective_model.auto_compact_threshold == (
            DEFAULT_AUTO_COMPACT_THRESHOLD
        )

    def test_a_deployments_own_threshold_wins_over_the_child_default(
        self, materialized_thresholds: list[int]
    ) -> None:
        config = _probe_config(
            limit=1_000_000, reserve=128_000, threshold=PRIOR_SHIPPED_THRESHOLD
        )

        launch = _resolve_child(config, committed_model=_committed_probe())

        assert materialized_thresholds == [CHILD_AUTO_COMPACT_THRESHOLD]
        assert launch.effective_model.auto_compact_threshold == PRIOR_SHIPPED_THRESHOLD

    def test_a_shipped_child_keeps_its_deployments_threshold(
        self, materialized_thresholds: list[int]
    ) -> None:
        raised = _resolve_child(build_test_vibe_config())
        assert raised.effective_model.auto_compact_threshold == SHIPPED_RAISE

        materialized_thresholds.clear()
        heavy = _resolve_child(build_test_vibe_config(active_model="@heavy"))

        assert heavy.effective_model.auto_compact_threshold == (PRIOR_SHIPPED_THRESHOLD)
        assert materialized_thresholds == [
            DEFAULT_AUTO_COMPACT_THRESHOLD,
            DEFAULT_AUTO_COMPACT_THRESHOLD,
        ]


class TestChildOverrideRejectedAtLoadIsLaunchSafe:
    """A child override that fails the invariant never fails a child launch."""

    def test_a_child_override_on_an_unknown_limit_deployment_rejects_the_file(
        self,
    ) -> None:
        snapshot = _deployments_snapshot({
            "glm-5-3": {"auto_compact_threshold": None, "context_limit": None}
        })

        with pytest.raises(ValidationError) as raised:
            _load(snapshot, auto_compact_threshold=RAISE_WITH_ROOM)

        message = str(raised.value)
        assert "mistral/zai-glm-5-3" in message
        assert "context limit is unknown" in message
        assert "context_limit" in message
        assert "max_completion_tokens" in message
        assert "revert the threshold raise" in message

    @pytest.mark.asyncio
    async def test_the_launch_proceeds_under_the_last_validated_value(self) -> None:
        state: dict[str, Any] = {"theme": "accepted"}
        session = OverridesLayer(data=state)
        unthresholded = _deployments_snapshot({
            "glm-5-3": {"auto_compact_threshold": None, "context_limit": None}
        })
        orchestrator = await ConfigOrchestrator.create(
            schema=ChartreuxConfigSchema,
            layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), session],
            default_layer_resolver=lambda: session,
            catalog_snapshot=unthresholded,
            catalog_loader=lambda: unthresholded,
        )
        accepted = orchestrator.config
        assert accepted.auto_compact_threshold == DEFAULT_AUTO_COMPACT_THRESHOLD
        assert (
            accepted.get_active_model().auto_compact_threshold
            == DEFAULT_AUTO_COMPACT_THRESHOLD
        )

        state["auto_compact_threshold"] = RAISE_WITH_ROOM
        with pytest.raises(ValueError, match="context limit is unknown"):
            await orchestrator.reload()

        retained = orchestrator.config
        assert retained is accepted
        launch = _resolve_child(retained)
        assert launch.effective_model.auto_compact_threshold == (
            DEFAULT_AUTO_COMPACT_THRESHOLD
        )
