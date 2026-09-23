"""Shipped model-catalog definitions.

Wire names were verified against the Mistral models API on 2026-09-16 and the
local Codex configuration/adapter. Prices are published list prices; genuinely
unknown values are represented by ``None``, never ``0.0``.
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
        },
        "gpt-6-astra": {
            "thinking": "low",
            "deployments": [
                {
                    "provider": "codex/local",
                    "name": "gpt-6-astra",
                    "prices": {"input": 10.0, "output": 50.0, "cached_input": 1.0},
                    "supports_images": True,
                    "auto_compact_threshold": 500000,
                }
            ],
        },
        "gpt-6-luna": {
            "thinking": "max",
            "deployments": [
                {
                    "provider": "codex/local",
                    "name": "gpt-6-luna",
                    "prices": {"input": 0.1, "output": 0.50, "cached_input": 0.01},
                    "supports_images": True,
                    "auto_compact_threshold": 200000,
                }
            ],
        },
        "gpt-6-sol": {
            "thinking": "medium",
            "deployments": [
                {
                    "provider": "codex/local",
                    "name": "gpt-6-sol",
                    "prices": {"input": 2.0, "output": 10.0, "cached_input": 0.2},
                    "supports_images": True,
                    "auto_compact_threshold": 500000,
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
            "models": ["gpt-6-luna"],
        },
        "large-worker": {
            "description": "strongest model for complex, high-stakes tasks",
            "models": ["gpt-6-sol", "glm-5-3"],
        },
        "small-reviewer": {
            "description": "fast model for focused reviews",
            "models": ["gpt-6-luna"],
        },
        "deep-reviewer": {
            "description": "strongest model for complex, high-stakes reviews",
            "models": ["gpt-6-astra", "glm-5-3"],
        },
    },
})
