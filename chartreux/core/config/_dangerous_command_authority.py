"""Immutable dangerous-command grant contributions; source validation is not runtime acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from chartreux.core.config.layer import ConfigLayer, RawConfig
from chartreux.core.config.models import DangerousCommandGrantsInput
from chartreux.core.dangerous_commands import (
    RemoteDestination,
    canonical_remote_destination,
)

GRANTS_FIELD = "dangerous_command_grants_by_project"
REMOTE_EXEC_GRANTS_FIELD = "remote_exec_grants_by_project"


@dataclass(frozen=True, slots=True)
class ProjectDangerousCommandGrant:
    project: Path
    operations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProjectRemoteExecGrant:
    project: Path
    destinations: tuple[RemoteDestination, ...]


def validate_grant_source(
    data: dict[str, Any], *, layer: ConfigLayer[RawConfig]
) -> tuple[ProjectDangerousCommandGrant, ...]:
    """Validate one loaded source before merging, including shadowed definitions.

    The caller must supply the actual installed layer (or its private staged copy),
    not reconstruct a user layer from a request's name/locator. Exact concrete
    types deliberately exclude custom layers and subclasses claiming user origin.
    Only persistable operation IDs are accepted: the registry marks every
    consequential operation non-persistable, so a saved grant can never carry one.
    No source reads, writes, cache publication, or environment loading occur here.
    Returned values may enter a ConfigCandidate, but only orchestrator acceptance
    makes that candidate usable by the runtime. Ordinary patches must separately
    preserve accepted grant contributions; being user-targeted is not user intent.
    """
    from chartreux.core.config.layers.default import DefaultConfigLayer
    from chartreux.core.config.layers.user import UserConfigLayer

    if GRANTS_FIELD not in data:
        return ()
    if type(layer) is DefaultConfigLayer and data[GRANTS_FIELD] == {}:
        return ()
    if type(layer) is not UserConfigLayer:
        raise ValueError(
            "dangerous_command_grants_by_project requires an actual user source "
            "(source: non-user)"
        )
    try:
        parsed = DangerousCommandGrantsInput.model_validate(data)
    except ValidationError:
        # Raise outside the handler: even exception context must not retain paths.
        pass
    else:
        return tuple(
            ProjectDangerousCommandGrant(Path(project), tuple(operations))
            for project, operations in sorted(
                parsed.dangerous_command_grants_by_project.items()
            )
        )
    raise ValueError(
        "Invalid dangerous_command_grants_by_project definition (source: user)"
    )


def validate_remote_exec_grant_source(
    data: dict[str, Any], *, layer: ConfigLayer[RawConfig]
) -> tuple[ProjectRemoteExecGrant, ...]:
    """Validate one loaded source's remote-exec destination grants.

    Same provenance rule as the operation grants: only an actual user source
    may carry the field, and canonicalization failures reject the source.
    Every destination is an exact ``user@host:port`` identity; wildcards and
    alias groups never validate, so a saved grant cannot name a pattern.
    """
    from chartreux.core.config.layers.default import DefaultConfigLayer
    from chartreux.core.config.layers.user import UserConfigLayer

    if REMOTE_EXEC_GRANTS_FIELD not in data:
        return ()
    if type(layer) is DefaultConfigLayer and data[REMOTE_EXEC_GRANTS_FIELD] == {}:
        return ()
    if type(layer) is not UserConfigLayer:
        raise ValueError(
            "remote_exec_grants_by_project requires an actual user source "
            "(source: non-user)"
        )
    try:
        parsed = DangerousCommandGrantsInput.model_validate(data)
    except ValidationError:
        # Raise outside the handler: even exception context must not retain
        # destination values.
        pass
    else:
        grants: list[ProjectRemoteExecGrant] = []
        for project, destinations in sorted(
            parsed.remote_exec_grants_by_project.items()
        ):
            parsed_destinations = tuple(
                canonical_remote_destination(item) for item in destinations
            )
            if any(destination is None for destination in parsed_destinations):
                raise ValueError(
                    "Invalid remote_exec_grants_by_project definition (source: user)"
                )
            grants.append(
                ProjectRemoteExecGrant(
                    Path(project),
                    tuple(
                        destination
                        for destination in parsed_destinations
                        if destination is not None
                    ),
                )
            )
        return tuple(grants)
    raise ValueError("Invalid remote_exec_grants_by_project definition (source: user)")
