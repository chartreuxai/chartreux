from __future__ import annotations

from pydantic import BaseModel
import pytest

from chartreux.core.tools.builtins.edit import EditArgs
from chartreux.core.tools.builtins.grep import GrepArgs
from chartreux.core.tools.builtins.read_file import ReadFileArgs
from chartreux.core.tools.builtins.write_file import WriteFileArgs

PATH_ARGUMENTS: list[tuple[type[BaseModel], str, dict[str, str]]] = [
    (WriteFileArgs, "file_path", {"content": "hello"}),
    (EditArgs, "file_path", {"old_string": "a", "new_string": "b"}),
    (ReadFileArgs, "file_path", {}),
    (GrepArgs, "path", {"pattern": "needle"}),
]
PATH_ARGUMENT_IDS = [model.__name__ for model, _, _ in PATH_ARGUMENTS]


@pytest.mark.parametrize(
    ("model", "field", "extra"), PATH_ARGUMENTS, ids=PATH_ARGUMENT_IDS
)
@pytest.mark.parametrize(
    "raw",
    [
        "/c/Users/acmedev/test/notes.md",
        "C:/Users/acmedev/test/notes.md",
        "/Users/acmedev/test/notes.md",
    ],
)
def test_path_argument_preserves_posix_and_foreign_input(
    model: type[BaseModel], field: str, extra: dict[str, str], raw: str
) -> None:
    args = model.model_validate({field: raw, **extra})
    assert getattr(args, field) == raw
