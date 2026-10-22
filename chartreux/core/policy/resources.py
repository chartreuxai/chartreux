from __future__ import annotations

from pathlib import Path
import re

from chartreux.core.policy.models import ResolvedResource, ResourceKind


class ResourceResolutionError(ValueError):
    pass


class ResourceLoader:
    MAX_BYTES = 65536
    _ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

    def __init__(
        self,
        directories: dict[ResourceKind, Path],
        builtins: dict[ResourceKind, dict[str, Path]] | None = None,
    ) -> None:
        self.directories = directories
        self.builtins = builtins or {}

    def resolve(self, kind: ResourceKind, reference: str) -> ResolvedResource:
        if not isinstance(reference, str) or not reference:
            raise ResourceResolutionError("resource id must be non-empty text")
        if reference == "none" and kind is ResourceKind.STRATEGY:
            return ResolvedResource(kind, "none", "builtin", None)
        if ":" in reference:
            source, name = reference.split(":", 1)
            if source not in {"builtin", "user"} or not name or ":" in name:
                raise ResourceResolutionError(
                    f"malformed resource qualification: {reference!r}"
                )
        else:
            name = reference
            source = "builtin" if name in self.builtins.get(kind, {}) else "user"
        if not self._ID.fullmatch(name) or name in {".", ".."}:
            raise ResourceResolutionError(f"invalid resource id: {reference!r}")
        shipped = self.builtins.get(kind, {})
        if source == "builtin":
            path = shipped.get(name)
            if path is None:
                raise ResourceResolutionError(f"unknown built-in {kind.value}: {name}")
        else:
            if name in shipped:
                raise ResourceResolutionError(f"reserved built-in name: {name}")
            path = self.directories[kind] / f"{name}.md"
        try:
            data = path.read_bytes()
            if len(data) > self.MAX_BYTES:
                raise ResourceResolutionError(
                    f"resource exceeds {self.MAX_BYTES} bytes"
                )
            content = data.decode("utf-8", errors="strict").strip()
        except ResourceResolutionError:
            raise
        except (OSError, UnicodeError) as exc:
            raise ResourceResolutionError(
                f"cannot read resource {reference!r}: {exc}"
            ) from exc
        if not content:
            raise ResourceResolutionError(f"resource is empty: {reference!r}")
        return ResolvedResource(kind, name, source, content, path)

    def validate_strategy(self, delegation: str, strategy: str) -> None:
        if delegation == "none" and strategy != "none":
            raise ResourceResolutionError(
                "identity delegation=none cannot select an orchestration strategy"
            )
        self.resolve(ResourceKind.STRATEGY, strategy)
