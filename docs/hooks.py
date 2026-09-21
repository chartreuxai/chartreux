from __future__ import annotations

from pathlib import Path
from shutil import copyfile
from typing import Any

from mkdocs.config.defaults import MkDocsConfig


def on_pre_build(config: MkDocsConfig, **kwargs: Any) -> None:
    """Copy canonical project documents into the build input."""
    root = Path(config.config_file_path).parent
    project_dir = Path(config.docs_dir) / "project"
    project_dir.mkdir(parents=True, exist_ok=True)

    for source_name, destination_name in (
        ("CHANGELOG.md", "changelog.md"),
        ("ACKNOWLEDGEMENTS.md", "acknowledgements.md"),
    ):
        source = root / source_name
        if source.is_file():
            copyfile(source, project_dir / destination_name)
