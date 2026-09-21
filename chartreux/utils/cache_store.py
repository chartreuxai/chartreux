from __future__ import annotations

import fcntl
import os
from pathlib import Path
import tempfile
import tomllib
from typing import Any, Protocol

import tomli_w

from chartreux.observability.logging import logger
from chartreux.utils.paths import get_chartreux_home

__all__ = ["CacheStore", "FileSystemCacheStore", "InMemoryCacheStore"]


class CacheStore(Protocol):
    def read_section(self, section: str) -> dict[str, Any]: ...

    def write_section(self, section: str, data: dict[str, Any]) -> None: ...


class InMemoryCacheStore:
    def __init__(self) -> None:
        self._sections: dict[str, dict[str, Any]] = {}

    def read_section(self, section: str) -> dict[str, Any]:
        return dict(self._sections.get(section, {}))

    def write_section(self, section: str, data: dict[str, Any]) -> None:
        self._sections.setdefault(section, {}).update(data)


class FileSystemCacheStore:
    def __init__(self, cache_path: Path | str | None = None) -> None:
        self._cache_path = (
            Path(cache_path)
            if cache_path is not None
            else get_chartreux_home() / "cache.toml"
        )

    def read_section(self, section: str) -> dict[str, Any]:
        data = self._read_cache().get(section)
        if not isinstance(data, dict):
            return {}
        return dict(data)

    def write_section(self, section: str, data: dict[str, Any]) -> None:
        temporary_path: Path | None = None
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self._cache_path.with_suffix(f"{self._cache_path.suffix}.lock")
            with lock_path.open("a+b") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                existing = self._read_cache()
                section_data = existing.get(section)
                if not isinstance(section_data, dict):
                    section_data = {}
                    existing[section] = section_data
                section_data.update(data)

                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=self._cache_path.parent, delete=False
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    tomli_w.dump(existing, temporary_file)
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())
                os.replace(temporary_path, self._cache_path)
                temporary_path = None
        except OSError:
            logger.debug(
                "Failed to write cache file %s", self._cache_path, exc_info=True
            )
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    logger.debug(
                        "Failed to remove temporary cache file %s",
                        temporary_path,
                        exc_info=True,
                    )

    def _read_cache(self) -> dict[str, Any]:
        try:
            with self._cache_path.open("rb") as f:
                return tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            return {}
