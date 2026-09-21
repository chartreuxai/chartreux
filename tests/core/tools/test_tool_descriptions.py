"""Tool descriptions are sourced from the sibling prompts/<tool>.md file and
exposed via get_full_description(); every argument carries a Field description.
"""

from __future__ import annotations

from chartreux.core.tools.builtins.edit import Edit
from chartreux.core.tools.builtins.read_file import ReadFile
from chartreux.core.tools.builtins.read_image import ReadImage
from chartreux.core.tools.builtins.web_fetch import WebFetch
from chartreux.core.tools.builtins.web_search import WebSearch
from chartreux.core.tools.builtins.write_file import WriteFile


def test_file_tools_use_unified_file_path_argument() -> None:
    for cls in (ReadFile, WriteFile, Edit):
        assert "file_path" in cls.get_parameters()["properties"]


def test_tool_names_are_unified() -> None:
    assert ReadFile.get_name() == "read_file"
    assert ReadImage.get_name() == "read_image"
    assert WriteFile.get_name() == "write_file"
    assert WebFetch.get_name() == "web_fetch"
    assert WebSearch.get_name() == "web_search"


def test_new_builtin_descriptions_load_sibling_prompts() -> None:
    for tool in (ReadImage, WebSearch):
        prompt = tool.get_tool_prompt()
        assert prompt is not None
        assert prompt == tool.get_full_description()
        assert prompt.strip()
