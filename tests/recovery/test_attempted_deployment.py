"""Review R11 — the failed attempt's deployment identity survives rollback.

After a failover, the coordinator rolls the committed model back to the
pre-attempt identity, so an envelope built from ``committed_model`` can pair
deployment A with the endpoint of failed attempt B. The fix freezes the
ATTEMPTED deployment on the raised error BEFORE the rollback
(``attach_attempted_deployment``, first-wins like ``attach_call_evidence``);
the observation carrier extracts it from the cause chain, and both the child
envelope builder and the root turn projection prefer it over the restored
committed identity.

Pinned here: the attachment contract (total, idempotent, first-wins), the
observation extraction through the gateway's error mapping (``raise ... from``
keeps the chain walkable), the root projection's preference, and the
coordinator-level failover A -> B where both attempts fail: the propagated
error carries B's deployment while the committed model is restored to A.

Honest labeling: every fault is a SYNTHETIC fixture on a controlled fake
backend — no live-provider reproduction (fail-closed egress).
"""

from __future__ import annotations

from types import MethodType

import httpx
import pytest

from chartreux.core.agent_loop._loop import AgentLoop
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import ModelConfig
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.recovery_models import (
    attach_attempted_deployment,
    build_root_turn_failure,
)
from chartreux.core.recovery_observation import observe_exception
from chartreux.core.session_types import CommittedModelIdentity
from tests.conftest import build_test_agent_loop
from tests.stubs.fake_backend import FakeBackend

_DEPLOYMENT_A = ("base", "test-first", "first")
_DEPLOYMENT_B = ("base", "test-second", "second")


class _ReadOnlyError(Exception):
    """An error whose ``attempted_deployment`` attribute rejects assignment
    (attachment stays total).
    """

    @property
    def attempted_deployment(self) -> None:
        return None


def _committed(identity: tuple[str, str, str]) -> CommittedModelIdentity:
    return CommittedModelIdentity(
        base_model=identity[0],
        provider=identity[1],
        wire_name=identity[2],
        catalog_revision="test-revision",
    )


def test_attach_attempted_deployment_is_total_idempotent_first_wins() -> None:
    error = RuntimeError("failed attempt")
    attach_attempted_deployment(error, _DEPLOYMENT_B)
    assert error.attempted_deployment == _DEPLOYMENT_B  # type: ignore[attr-defined]

    # First wins: a second attachment never overwrites the measurement.
    attach_attempted_deployment(error, _DEPLOYMENT_A)
    assert error.attempted_deployment == _DEPLOYMENT_B  # type: ignore[attr-defined]

    # Total: an error that rejects attribute assignment never raises.
    read_only = _ReadOnlyError("read only")
    attach_attempted_deployment(read_only, _DEPLOYMENT_A)
    assert read_only.attempted_deployment is None


def test_observe_exception_extracts_attempted_deployment_from_chain() -> None:
    """The gateway maps errors via ``raise mapped from original``; the
    observation's cause walk must still find the attached identity.
    """
    original = httpx.ReadTimeout("second slow")
    attach_attempted_deployment(original, _DEPLOYMENT_B)
    mapped = RuntimeError("API error from test-second (model: second)")
    mapped.__cause__ = original

    observation = observe_exception(mapped, phase="turn", turn_id="t")
    assert observation.attempted_deployment == _DEPLOYMENT_B

    # Control: no attachment anywhere in the chain -> unavailable, never guessed.
    plain = observe_exception(RuntimeError("plain"), phase="turn", turn_id="t")
    assert plain.attempted_deployment is None


def test_root_turn_failure_prefers_attached_attempted_deployment() -> None:
    """R11: the root projection reports the deployment that served the failed
    attempt, not the identity the rollback restored.
    """
    error = httpx.ReadTimeout("second slow")
    attach_attempted_deployment(error, _DEPLOYMENT_B)
    observation = observe_exception(error, phase="turn", turn_id="turn-1")

    failure = build_root_turn_failure(
        observation=observation, turn_id="turn-1", committed=_committed(_DEPLOYMENT_A)
    )
    assert failure.attempted_base_model == "base"
    assert failure.attempted_provider == "test-second"
    assert failure.attempted_wire_name == "second"

    # Control: no attachment -> the committed identity is reported.
    plain = build_root_turn_failure(
        observation=observe_exception(
            httpx.ReadTimeout("slow"), phase="turn", turn_id="t"
        ),
        turn_id="t",
        committed=_committed(_DEPLOYMENT_A),
    )
    assert plain.attempted_base_model == "base"
    assert plain.attempted_provider == "test-first"
    assert plain.attempted_wire_name == "first"


# ---------------------------------------------------------------------------
# Coordinator level: failover A -> B, both attempts fail
# ---------------------------------------------------------------------------


def _snapshot() -> CatalogSnapshot:
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test-first": {"api_base": "https://first.invalid"},
                "test-second": {"api_base": "https://second.invalid"},
            },
            "models": {
                "base": {
                    "deployments": [
                        {"provider": "test-first", "name": "first"},
                        {"provider": "test-second", "name": "second"},
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


def _failover_backends(
    agent: AgentLoop, first: FakeBackend, second: FakeBackend
) -> None:
    def backend_for_attempt(
        _self: AgentLoop, model: ModelConfig, _budget: object
    ) -> FakeBackend:
        return {"test-first": first, "test-second": second}[model.provider]

    agent._backend_for_attempt = MethodType(backend_for_attempt, agent)  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_failed_failover_carries_attempted_not_restored_deployment() -> None:
    """R11 (coordinator level): after failover A -> B with both attempts
    failing, the propagated error carries B's deployment identity while the
    committed model is rolled back to A — the envelope must not pair A with
    B's endpoint.
    """
    agent = _agent()
    original_identity = agent.committed_model
    assert original_identity is not None
    assert (original_identity.provider, original_identity.wire_name) == (
        "test-first",
        "first",
    )
    _failover_backends(
        agent,
        FakeBackend(exception_to_raise=httpx.ConnectError("first down")),
        FakeBackend(exception_to_raise=httpx.ReadTimeout("second slow")),
    )

    with pytest.raises(RuntimeError, match="API error") as caught:
        await agent._chat()

    # The committed identity is restored to the pre-attempt deployment A...
    assert agent.committed_model == original_identity
    # ...but the failed attempt's identity is frozen on the raised error: B.
    assert getattr(caught.value, "attempted_deployment", None) == _DEPLOYMENT_B
    # The observation carrier (what the envelope builder consumes) sees B too.
    observation = observe_exception(caught.value, phase="turn", turn_id="t")
    assert observation.attempted_deployment == _DEPLOYMENT_B

    await agent.aclose()
