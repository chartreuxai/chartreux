from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from chartreux.utils.io import read_safe


class ExternalEditor:
    """Handles opening an external editor to edit prompt content."""

    @staticmethod
    def get_editor() -> str:
        return os.environ.get("VISUAL") or os.environ.get("EDITOR") or "nano"

    @classmethod
    def resolve_command(cls) -> list[str]:
        """Validate the editor command before the app suspends.

        Raises ``ValueError`` for malformed quoting or an empty command, and
        ``OSError`` when the executable cannot be resolved.
        """
        parts = shlex.split(cls.get_editor())
        if not parts:
            raise ValueError("Editor command is empty")
        if shutil.which(parts[0]) is None:
            raise OSError(f"Editor not found: {parts[0]}")
        return parts

    @classmethod
    def edit_file(cls, file_path: Path, *, check: bool = False) -> None:
        editor = cls.get_editor()
        parts = shlex.split(editor)
        # Trusted user-operated editor: intentionally retains the user's environment.
        subprocess.run([*parts, str(file_path)], check=check)

    def edit(self, initial_content: str = "") -> str | None:
        filepath: str | None = None
        try:
            fd, filepath = tempfile.mkstemp(suffix=".md", prefix="vibe_")
            with os.fdopen(fd, "w") as f:
                f.write(initial_content)

            self.edit_file(Path(filepath), check=True)

            content = read_safe(Path(filepath)).text.rstrip()
            return content if content != initial_content else None
        except (OSError, ValueError, subprocess.CalledProcessError):
            return
        finally:
            if filepath is not None:
                try:
                    os.unlink(filepath)
                except OSError:
                    pass
