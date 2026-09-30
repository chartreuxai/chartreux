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
            "temperature": 0.2,
            "deployments": [
                {
                    "provider": "mistral",
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
            "description": "main assistant default model and thinking level",
            "model": "glm-5-3",
            "thinking": "high",
        },
        "large": {
            "description": "capacity preset for complex tasks",
            "model": "glm-5-3",
            "thinking": "high",
        },
        "medium": {
            "description": "capacity preset for routine tasks",
            "model": "glm-5-3",
            "thinking": "medium",
        },
        "small": {
            "description": "capacity preset for focused tasks",
            "model": "glm-5-3",
            "thinking": "low",
        },
    },
})
