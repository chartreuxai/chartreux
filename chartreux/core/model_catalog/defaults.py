"""Shipped model-catalog definitions.

The shipped catalog is neutral and publicly reachable only: it names the
Mistral public API and models Mistral actually serves (wire names re-verified
against the Mistral models API on 2026-09-26). Personal setups — local
proxies, LAN endpoints, private model pins — belong in the user overlay at
``$CHARTREUX_HOME/models.toml``, never here. Prices are published list
prices, except the shipped ``mistral-large-4`` entry, which carries
launch-sale prices below the published list; genuinely unknown values are
represented by ``None``, never ``0.0``.

``context_limit`` and ``max_completion_tokens`` are provider-documented
deployment facts (the Mistral model cards), never policy values: they stay
``None`` unless the provider documentation states them, and they never feed
``auto_compact_threshold``.

A shipped ``auto_compact_threshold`` above 400k is gated by the measured
headroom invariant at config load and at launch: every shipped deployment
receiving the 500k raise ships with a KNOWN context limit AND a KNOWN output
reserve, and a deployment that documents either fact as unknown keeps its prior
threshold. A user overlay raising a threshold on an unknown-limit deployment
must populate the limit field, or the overlay's file fails validation at load —
nothing silently raises, nothing silently stays.
"""

from __future__ import annotations

from chartreux.core.model_catalog.schema import DeploymentDefinition, ModelCatalog

SHIPPED_CATALOG = ModelCatalog.model_validate({
    "providers": {
        "mistral": {
            "api_base": "https://api.mistral.ai/v1",
            "api_key_env_var": "MISTRAL_API_KEY",
            "api_style": "openai",
            "backend": "mistral",
            "reasoning_field_name": "reasoning_content",
            "emits_finish_reason": True,
        }
    },
    "models": {
        "glm-5-3": {
            "thinking": "high",
            "deployments": [
                {
                    "provider": "mistral",
                    "name": "zai-glm-5-3",
                    "prices": {"input": 1.4, "output": 4.4, "cached_input": 0.14},
                    "supports_images": False,
                    # Mistral model card for zai-glm-5-3: 1M context window,
                    # 128k maximum output. 500k + 217k measured one-turn growth
                    # allowance = 717k inside 1M - 128k, so the raise holds.
                    "auto_compact_threshold": 500000,
                    "context_limit": 1000000,
                    "max_completion_tokens": 128000,
                }
            ],
        },
        "mistral-large-4": {
            "thinking": "high",
            "deployments": [
                {
                    "provider": "mistral",
                    "name": "mistral-large-4",
                    "prices": {"input": 0.68, "output": 2.09, "cached_input": 0.07},
                    "supports_images": True,
                    # The Mistral model card documents the 1M context window but
                    # no maximum output, so this deployment ships no reserve and
                    # KEEPS its prior threshold: the raise fails closed without a
                    # documented reserve.
                    "auto_compact_threshold": 400000,
                    "context_limit": 1000000,
                    "max_completion_tokens": None,
                }
            ],
        },
    },
    "roles": {
        "main": {
            "description": "main assistant preset",
            "model": "glm-5-3",
            "thinking": "high",
        },
        "worker": {
            "description": "worker preset",
            "model": "glm-5-3",
            "thinking": "medium",
        },
        "scout": {"description": "scout preset", "model": "glm-5-3", "thinking": "low"},
        "heavy": {
            "description": "heavy preset",
            "model": "mistral-large-4",
            "thinking": "high",
        },
    },
})


#: The compaction threshold a bounded child runs under when its deployment
#: documents no threshold of its own. Unlike ``context_limit`` and
#: ``max_completion_tokens`` this is a POLICY value, so it is gated exactly like
#: the shipped raise (see :func:`child_auto_compact_threshold`).
CHILD_AUTO_COMPACT_THRESHOLD = 500_000


def child_auto_compact_threshold(
    deployment: DeploymentDefinition, fallback: int
) -> int:
    """The child default for *deployment*, gated by the threshold invariant.

    The child default raises to :data:`CHILD_AUTO_COMPACT_THRESHOLD` only where
    the deployment documents a context limit AND an output reserve AND the
    raised value still leaves the measured one-turn growth allowance inside
    ``limit - reserve``. Every other deployment keeps its prior child default
    (*fallback*: the configured ``auto_compact_threshold``, which config load
    validated), so the gate never runs arithmetic on an unknown fact.
    """
    from chartreux.core.config.chartreux_schema import ONE_TURN_GROWTH_ALLOWANCE

    limit = deployment.context_limit
    reserve = deployment.max_completion_tokens
    if limit is None or reserve is None:
        return fallback
    if CHILD_AUTO_COMPACT_THRESHOLD + ONE_TURN_GROWTH_ALLOWANCE > limit - reserve:
        return fallback
    return max(fallback, CHILD_AUTO_COMPACT_THRESHOLD)
