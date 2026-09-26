"""Surfacing of AllDeploymentsUnavailableError from a real turn.

The error is raised by the availability resolver when every deployment of the
committed base is excluded, and nothing in production catches it: it must reach
the protocol layer as a dedicated, user-facing error instead of a generic
internal failure.
"""

from __future__ import annotations

from typing import cast

import pytest

from chartreux.app_server._utils import public_error
from chartreux.app_server.models import TurnErrorCode
from chartreux.core.agent_loop._loop import AgentLoop
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.model_catalog.availability import AllDeploymentsUnavailableError
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from tests.conftest import build_test_agent_loop
from tests.stubs.fake_backend import FakeBackend


def _snapshot() -> CatalogSnapshot:
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test/first": {"api_base": "https://first.invalid"},
                "test/second": {"api_base": "https://second.invalid"},
            },
            "models": {
                "base": {
                    "deployments": [
                        {"provider": "test/first", "name": "first"},
                        {"provider": "test/second", "name": "second"},
                    ]
                }
            },
            "roles": {},
        }),
        "test-revision",
    )


def _agent() -> AgentLoop:
    snapshot = _snapshot()
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "base"}, context={"catalog_snapshot": snapshot}
    ).attach_catalog_snapshot(snapshot)
    return build_test_agent_loop(config=config, backend=FakeBackend())


@pytest.mark.asyncio
async def test_all_deployments_unavailable_turn_surfaces_dedicated_error() -> None:
    agent = _agent()
    registry = agent.config_orchestrator.availability_registry
    # Every deployment of the committed base is in cooldown, so the turn has
    # no eligible candidate left.
    registry.record_failure("base", "test/first")
    registry.record_failure("base", "test/second")

    with pytest.raises(AllDeploymentsUnavailableError) as exc_info:
        async for _ in agent.act("Hello"):
            pass

    surfaced = public_error(exc_info.value)
    assert surfaced.code == TurnErrorCode.ALL_DEPLOYMENTS_UNAVAILABLE
    # The user-facing message names the model and each deployment's exclusion
    # reason, rather than surfacing as a generic internal ValueError.
    assert "'base'" in surfaced.message
    assert "test/first" in surfaced.message
    assert "test/second" in surfaced.message
    assert surfaced.message.count("cooldown") == 2
    details = surfaced.details
    assert isinstance(details, dict)
    assert details["base_model"] == "base"
    exclusions = cast("list[dict[str, str]]", details["exclusions"])
    assert {exclusion["reason"] for exclusion in exclusions} == {"cooldown"}
    assert {exclusion["provider"] for exclusion in exclusions} == {
        "test/first",
        "test/second",
    }
