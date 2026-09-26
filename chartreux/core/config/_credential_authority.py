"""Source authority for credential environment passthrough."""

from __future__ import annotations

from typing import Any

from chartreux.core.config.layer import ConfigLayer, RawConfig

CREDENTIAL_ENV_FIELD = "credential_env_passthrough"


def validate_credential_env_source(
    data: dict[str, Any], *, layer: ConfigLayer[RawConfig]
) -> None:
    """Reject every non-user contribution before merging, even if shadowed."""
    from chartreux.core.config.layers.default import DefaultConfigLayer
    from chartreux.core.config.layers.user import UserConfigLayer

    if CREDENTIAL_ENV_FIELD not in data:
        return
    if type(layer) is UserConfigLayer:
        return
    if type(layer) is DefaultConfigLayer and data[CREDENTIAL_ENV_FIELD] == []:
        return
    raise ValueError(
        "credential_env_passthrough requires an actual user source (source: non-user)"
    )
