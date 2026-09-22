"""Shipped model-catalog definitions.

Wire names were verified against the Mistral models API on 2026-09-16 and the
local Codex configuration/adapter. Prices are unknown unless independently
verified; unknown is deliberately represented by ``None``, never ``0.0``.
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
        },
        "codex/local": {
            "api_base": "http://127.0.0.1:18080/v1",
            "api_style": "openai-responses",
            "backend": "generic",
        },
    },
    "models": {
        "glm-5-3": {
            "thinking": "medium",
            "temperature": 0.2,
            "deployments": [
                {
                    "provider": "mistral/default",
                    "name": "zai-glm-5-3",
                    "supports_images": False,
                }
            ],
        },
        "gpt-6-astra": {
            "thinking": "low",
            "deployments": [
                {
                    "provider": "codex/local",
                    "name": "gpt-6-astra",
                    "supports_images": True,
                }
            ],
        },
        "gpt-5.6-luna": {
            "thinking": "high",
            "deployments": [
                {
                    "provider": "codex/local",
                    "name": "gpt-5.6-luna",
                    "supports_images": True,
                }
            ],
        },
        "gpt-5.6-sol": {
            "thinking": "medium",
            "deployments": [
                {
                    "provider": "codex/local",
                    "name": "gpt-5.6-sol",
                    "supports_images": True,
                }
            ],
        },
        "gpt-5.6-terra": {
            "thinking": "medium",
            "deployments": [
                {
                    "provider": "codex/local",
                    "name": "gpt-5.6-terra",
                    "supports_images": True,
                }
            ],
        },
    },
    "roles": {
        "orchestrator": {
            "description": "primary model for orchestration and coordination",
            "models": ["glm-5-3"],
        },
        "advisor": {
            "description": "independent perspective for architectural guidance",
            "models": ["gpt-6-astra"],
        },
        "small-worker": {
            "description": "fast model for focused implementation tasks",
            "models": ["gpt-5.6-luna"],
        },
        "medium-worker": {
            "description": "capable model for general implementation tasks",
            "models": ["gpt-5.6-terra"],
        },
        "large-worker": {
            "description": "strongest model for complex, high-stakes tasks",
            "models": ["gpt-5.6-sol", "glm-5-3"],
        },
        "small-reviewer": {
            "description": "fast model for focused reviews",
            "models": ["gpt-5.6-luna"],
        },
        "medium-reviewer": {
            "description": "capable model for general reviews",
            "models": ["gpt-5.6-terra"],
        },
        "deep-reviewer": {
            "description": "strongest model for complex, high-stakes reviews",
            "models": ["gpt-6-astra", "glm-5-3"],
        },
    },
})
