from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from chartreux.core.paths import dedup_paths

if TYPE_CHECKING:
    from chartreux.core.config._restrictions import SourceRestrictions


@dataclass(frozen=True)
class Workspace:
    """Canonical session position and explicit read/write authority.

    Discovery roots are not grants. Runtime-owned callers may supply additional
    roots only after validating their user-source authority outside config merges.
    This is an accident guard, not protection against filesystem races.
    """

    cwd: Path
    authorized_roots: tuple[Path, ...]
    ceiling: Workspace | None = None

    @classmethod
    def from_restrictions(
        cls,
        cwd: Path,
        sources: Iterable[SourceRestrictions],
        *,
        ceiling: Workspace | None = None,
    ) -> Workspace:
        """Consume accepted canonical identities without resolving them again.

        Resolving an accepted root here would silently grant a replacement symlink
        target. Only the exact current project key contributes related roots.
        """
        roots = (cwd,) + tuple(
            root
            for source in sources
            for authority in source.authorized_roots
            if authority.project == cwd
            for root in authority.roots
        )
        return cls(cwd, tuple(dict.fromkeys(roots)), ceiling)

    @classmethod
    def for_session(
        cls, cwd: Path, *, authorized_roots: Iterable[Path] = ()
    ) -> Workspace:
        """Build the workspace a session starts with.

        The working directory is authorised whether or not it is trusted or
        listed as a project root, which is what keeps an unconfigured or
        untrusted directory usable.
        """
        resolved = cwd.resolve()
        return cls(
            cwd=resolved,
            authorized_roots=tuple(dedup_paths([resolved, *authorized_roots])),
        )

    def allows(self, resolved_path: Path) -> bool:
        """Check a canonical target without redefining replaced root identities."""
        try:
            target = resolved_path.resolve()
            return (self.ceiling is None or self.ceiling.allows(target)) and any(
                root.resolve() == root and target.is_relative_to(root)
                for root in self.authorized_roots
            )
        except (ValueError, OSError):
            return False
