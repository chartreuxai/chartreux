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
        "glm-5-2": {
            "aliases": (),
            "thinking": "high",
            "temperature": 0.2,
            "deployments": [
                {
                    "provider": "mistral/default",
                    "name": "glm-5-2",
                    "supports_images": False,
                    "auto_compact_threshold": 400000,
                }
            ],
        },
        "glm-5-3": {
            "aliases": ("zai-glm-5-3", "zai-glm-5", "zai-glm-latest"),
            "thinking": "medium",
            "deployments": [
                {
                    "provider": "mistral/default",
                    "name": "zai-glm-5-3",
                    "supports_images": False,
                }
            ],
        },
        "mistral-small": {
            "aliases": ("mistral-small-latest",),
            "thinking": "high",
            "deployments": [
                {
                    "provider": "mistral/default",
                    "name": "mistral-small-latest",
                    "supports_images": False,
                }
            ],
        },
        "gpt-6-astra": {
            "aliases": (),
            "thinking": "medium",
            "deployments": [
                {
                    "provider": "codex/local",
                    "name": "gpt-6-astra",
                    "supports_images": True,
                }
            ],
        },
        "gpt-5.6-luna": {
            "aliases": (),
            "thinking": "medium",
            "deployments": [
                {
                    "provider": "codex/local",
                    "name": "gpt-5.6-luna",
                    "supports_images": True,
                }
            ],
        },
        "gpt-5.6-sol": {
            "aliases": (),
            "thinking": "low",
            "deployments": [
                {
                    "provider": "codex/local",
                    "name": "gpt-5.6-sol",
                    "supports_images": True,
                }
            ],
        },
        "gpt-5.6-terra": {
            "aliases": (),
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
    "tags": {},
})
