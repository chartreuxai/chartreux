"""Shipped model-catalog definitions.

The shipped catalog is neutral and publicly reachable only: it names the
Mistral public API and models Mistral actually serves (wire names re-verified
against the Mistral models API on 2026-09-26). Personal setups — local
proxies, LAN endpoints, private model pins — belong in the user overlay at
``$CHARTREUX_HOME/models.toml``, never here. Prices are published list
prices; genuinely unknown values are represented by ``None``, never ``0.0``.
"""

from __future__ import annotations

from chartreux.core.model_catalog.schema import ModelCatalog

SHIPPED_CATALOG = ModelCatalog.model_validate({
    "providers": {
        "mistral/default": {
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
            "temperature": 0.2,
            "deployments": [
                {
                    "provider": "mistral/default",
                    "name": "zai-glm-5-3",
                    "prices": {"input": 1.4, "output": 4.4, "cached_input": 0.14},
                    "supports_images": False,
                    "auto_compact_threshold": 400000,
                }
            ],
        }
    },
    "roles": {
        "orchestrator": {
            "description": "primary model for orchestration and coordination",
            "models": ["glm-5-3"],
        },
        "advisor": {
            "description": "independent perspective for architectural guidance",
            "models": ["glm-5-3"],
        },
        "small-worker": {
            "description": "fast model for focused implementation tasks",
            "models": ["glm-5-3"],
        },
        "large-worker": {
            "description": "strongest model for complex, high-stakes tasks",
            "models": ["glm-5-3"],
        },
        "small-reviewer": {
            "description": "fast model for focused reviews",
            "models": ["glm-5-3"],
        },
        "deep-reviewer": {
            "description": "strongest model for complex, high-stakes reviews",
            "models": ["glm-5-3"],
        },
    },
})
