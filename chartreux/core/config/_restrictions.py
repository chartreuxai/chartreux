"""Immutable source contributions, not a policy service or an acceptance API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chartreux.core.config._root_authority import (
    ROOTS_FIELD,
    ProjectRootAuthority,
    validate_root_source,
)
from chartreux.core.config.fingerprint import create_dict_fingerprint
from chartreux.core.config.layer import ConfigLayer, RawConfig
from chartreux.core.config.schema import ConfigSchema
from chartreux.core.tools.models import ToolPermission


class _ToolRestrictionInput(BaseModel):
    # Validate only restriction-bearing fields here; the tool's own schema still
    # validates its complete effective config at the existing manager boundary.
    model_config = ConfigDict(extra="ignore")

    permission: ToolPermission = ToolPermission.ASK
    denylist: tuple[str, ...] = ()
    sensitive_patterns: tuple[str, ...] = ()


class _RestrictionInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tools: dict[str, _ToolRestrictionInput] = Field(default_factory=dict)


class _ReplacementToolInput(_ToolRestrictionInput):
    model_config = ConfigDict(extra="forbid")


class _ReplacementInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools: dict[str, _ReplacementToolInput]


@dataclass(frozen=True, slots=True)
class ToolRestriction:
    tool_name: str
    denied: bool
    denylist: tuple[str, ...]
    sensitive_patterns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PolicySourceIdentity:
    """Opaque in-memory contribution identity; unrelated to paths or values."""

    owner: UUID
    source: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True)
class SourceRestrictions:
    layer_name: str
    locator: str
    kind: Literal["source", "mode"]
    content_fingerprint: str
    store_fingerprint: str | None
    tools: tuple[ToolRestriction, ...]
    identity: PolicySourceIdentity | None = None
    authorized_roots: tuple[ProjectRootAuthority, ...] = ()
    # An empty explicit overlay revokes persisted roots; a tool-only projection
    # must leave the backing source's roots alone.
    replaces_roots: bool = False

    def project_roots(self) -> dict[str, list[str]]:
        """Return an independent canonical root map for private runtime staging."""
        return {
            str(item.project): [str(root) for root in item.roots]
            for item in self.authorized_roots
        }

    def replace_in(self, data: dict[str, Any]) -> dict[str, Any]:
        """Replace this source's complete restriction projection, not its settings."""
        import copy

        result = copy.deepcopy(data)
        tools = result.setdefault("tools", {})
        for settings in tools.values():
            if settings.get("permission") == ToolPermission.NEVER:
                settings["permission"] = ToolPermission.ASK.value
            settings.pop("denylist", None)
            settings.pop("sensitive_patterns", None)
        for restriction in self.tools:
            settings = tools.setdefault(restriction.tool_name, {})
            if restriction.denied:
                settings["permission"] = ToolPermission.NEVER.value
            settings["denylist"] = list(restriction.denylist)
            settings["sensitive_patterns"] = list(restriction.sensitive_patterns)
        if self.replaces_roots:
            result[ROOTS_FIELD] = self.project_roots()
        return result

    def preserves(self, previous: SourceRestrictions) -> bool:
        """Conservative syntactic check; generic patches cannot revoke authority."""
        if self.authorized_roots != previous.authorized_roots:
            return False
        tools = {tool.tool_name: tool for tool in self.tools}
        for old in previous.tools:
            new = tools.get(old.tool_name)
            if new is None or (old.denied and not new.denied):
                return False
            if not set(old.denylist) <= set(new.denylist):
                return False
            if not set(old.sensitive_patterns) <= set(new.sensitive_patterns):
                return False
        return True

    @classmethod
    def from_raw(
        cls,
        data: dict[str, Any],
        *,
        layer_name: str,
        locator: str,
        kind: Literal["source", "mode"],
        store_fingerprint: str | None,
        replacement: bool = False,
        identity: PolicySourceIdentity | None = None,
        root_layer: ConfigLayer[RawConfig] | None = None,
        replaces_roots: bool = False,
    ) -> SourceRestrictions:
        # Never infer authority from a source name, even for an empty assertion.
        if root_layer is None:
            if ROOTS_FIELD in data:
                raise ValueError(
                    "Authorized root source provenance unavailable: "
                    "authorized_roots_by_project (source: unknown)"
                )
            authorized_roots = ()
        else:
            authorized_roots = validate_root_source(data, layer=root_layer)
        try:
            if replacement:
                _ReplacementInput.model_validate(data)
            parsed = _RestrictionInput.model_validate(data)
        except ValidationError as exc:
            # Do not echo arbitrary source values (which may contain secrets).
            fields = ", ".join(
                ".".join(str(part) for part in error["loc"])
                for error in exc.errors(include_input=False)
            )
            raise ValueError(
                f"Invalid restrictions in {layer_name} ({locator}): {fields}"
            ) from None
        return cls(
            layer_name=layer_name,
            identity=identity,
            authorized_roots=authorized_roots,
            replaces_roots=replaces_roots,
            locator=locator,
            kind=kind,
            content_fingerprint=create_dict_fingerprint(data),
            store_fingerprint=store_fingerprint,
            tools=tuple(
                ToolRestriction(
                    name,
                    tool.permission == ToolPermission.NEVER,
                    tool.denylist,
                    tool.sensitive_patterns,
                )
                for name, tool in parsed.tools.items()
                if tool.permission == ToolPermission.NEVER
                or tool.denylist
                or tool.sensitive_patterns
            ),
        )


def partition_policy_sources(
    restrictions: tuple[SourceRestrictions, ...], *, owner: UUID
) -> tuple[tuple[SourceRestrictions, ...], tuple[SourceRestrictions, ...]]:
    """Extract own/inherited contributions without subtracting policy values.

    Copies retain identity; equal explicit assertions by different owners do not.
    Unknown legacy provenance blocks tree revocation rather than guessing ownership.
    A registry can replace/remove one contribution by its complete ``identity``;
    neither layer name nor locator alone is a tree-wide source key.
    """
    own: list[SourceRestrictions] = []
    inherited: list[SourceRestrictions] = []
    for restriction in restrictions:
        if restriction.identity is None:
            raise ValueError(
                "Policy source provenance unavailable; tree revocation blocked"
            )
        target = own if restriction.identity.owner == owner else inherited
        target.append(restriction)
    return tuple(own), tuple(inherited)


@dataclass(frozen=True, slots=True)
class ConfigCandidate[S: ConfigSchema]:
    """Prepared data only. Holding a candidate never makes it accepted authority.

    Restriction values are deeply immutable and independent of the ordinary
    schema's nested containers. Its content fingerprint is an in-memory source
    revision, not a serialized config format version or proof of user intent.
    """

    config: S
    restrictions: tuple[SourceRestrictions, ...]
