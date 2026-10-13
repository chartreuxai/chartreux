from __future__ import annotations

from pathlib import Path

from chartreux.core.file_outline import (
    MAX_OUTLINE_DEPTH,
    MAX_OUTLINE_RENDER_BYTES,
    MAX_OUTLINE_SOURCE_BYTES,
    MAX_OUTLINE_SYMBOLS,
    SYMBOL_KIND_ASSIGNMENT,
    SYMBOL_KIND_CLASS,
    SYMBOL_KIND_FUNCTION,
    FileOutline,
    OutlineSymbol,
    outline_file,
    outline_source,
)

SYMBOLS_SOURCE = """\
import os


class Greeter:
    def greet(self, name):
        return f"hi {name}"

    async def agreet(self):
        return "hi"


def top_level(a, b):
    def inner():
        return a

    return inner


CONST = 1
make = lambda: None
factory = build_thing()
alias = os.path
"""

MULTILINE_SOURCE = """\
def multiline(
    a,
    b,
):
    return (
        a
        + b
    )
"""

NESTED_SOURCE = """\
class Outer:
    class Inner:
        def method(self):
            pass
"""

DECORATED_SOURCE = """\
@register
def decorated():
    pass
"""

COMPOUND_SOURCE = """\
import sys

if sys.version_info >= (3, 12):
    def modern():
        pass
else:
    def legacy():
        pass

try:
    import json
except ImportError:
    def fallback():
        pass

CONST = 1
"""


def _tuples(outline: FileOutline) -> list[tuple[str, str, int, int]]:
    return [(s.name, s.kind, s.line, s.end_line) for s in outline.symbols]


class TestSymbolExtraction:
    def test_classes_methods_functions_with_current_line_ranges(self) -> None:
        outline = outline_source(SYMBOLS_SOURCE)
        assert outline is not None
        assert _tuples(outline) == [
            ("Greeter", SYMBOL_KIND_CLASS, 4, 9),
            ("Greeter.greet", SYMBOL_KIND_FUNCTION, 5, 6),
            ("Greeter.agreet", SYMBOL_KIND_FUNCTION, 8, 9),
            ("top_level", SYMBOL_KIND_FUNCTION, 12, 16),
            ("top_level.inner", SYMBOL_KIND_FUNCTION, 13, 14),
            ("make", SYMBOL_KIND_ASSIGNMENT, 20, 20),
            ("factory", SYMBOL_KIND_ASSIGNMENT, 21, 21),
            ("alias", SYMBOL_KIND_ASSIGNMENT, 22, 22),
        ]
        assert outline.line_count == 22
        assert outline.truncated is False

    def test_multiline_definition_uses_end_lineno(self) -> None:
        outline = outline_source(MULTILINE_SOURCE)
        assert outline is not None
        assert _tuples(outline) == [("multiline", SYMBOL_KIND_FUNCTION, 1, 8)]

    def test_nested_classes_get_qualified_names(self) -> None:
        outline = outline_source(NESTED_SOURCE)
        assert outline is not None
        assert _tuples(outline) == [
            ("Outer", SYMBOL_KIND_CLASS, 1, 4),
            ("Outer.Inner", SYMBOL_KIND_CLASS, 2, 4),
            ("Outer.Inner.method", SYMBOL_KIND_FUNCTION, 3, 4),
        ]

    def test_decorated_definition_lines_point_at_the_def(self) -> None:
        outline = outline_source(DECORATED_SOURCE)
        assert outline is not None
        assert _tuples(outline) == [("decorated", SYMBOL_KIND_FUNCTION, 2, 3)]

    def test_module_level_compound_statements_are_flattened_in_order(self) -> None:
        outline = outline_source(COMPOUND_SOURCE)
        assert outline is not None
        # Source order across if/else and try/except bodies; the constant
        # assignments (``json = None``, ``CONST = 1``) are excluded as noise.
        assert _tuples(outline) == [
            ("modern", SYMBOL_KIND_FUNCTION, 4, 5),
            ("legacy", SYMBOL_KIND_FUNCTION, 7, 8),
            ("fallback", SYMBOL_KIND_FUNCTION, 13, 14),
        ]

    def test_annotated_assignment_of_callable_is_outlined(self) -> None:
        outline = outline_source("handler: Callable = make_handler\n")
        assert outline is not None
        assert _tuples(outline) == [("handler", SYMBOL_KIND_ASSIGNMENT, 1, 1)]

    def test_constants_and_inner_assignments_are_excluded(self) -> None:
        source = "X = 1\n\n\ndef f():\n    local = lambda: None\n    return local\n"
        outline = outline_source(source)
        assert outline is not None
        assert _tuples(outline) == [("f", SYMBOL_KIND_FUNCTION, 4, 6)]

    def test_bytes_source_and_coding_cookie(self) -> None:
        source = b"# -*- coding: latin-1 -*-\ncaf\xe9 = lambda: None\n"
        outline = outline_source(source)
        assert outline is not None
        assert [s.name for s in outline.symbols] == ["ca" + "f\xe9"]


class TestDegradation:
    def test_syntax_error_degrades_to_none(self) -> None:
        assert outline_source("def broken(:\n    pass\n") is None

    def test_non_python_path_degrades_to_none(self) -> None:
        assert outline_source("hello\n", path="notes.txt") is None
        assert outline_source("hello\n", path="README") is None

    def test_oversized_source_degrades_to_none(self) -> None:
        source = b"x = 1\n" * (MAX_OUTLINE_SOURCE_BYTES // 6 + 1)
        assert len(source) > MAX_OUTLINE_SOURCE_BYTES
        assert outline_source(source) is None

    def test_empty_source_yields_an_empty_outline(self) -> None:
        outline = outline_source("")
        assert outline is not None
        assert outline.symbols == []
        assert outline.truncated is False
        assert outline.line_count == 0
        assert outline.render() == ""

    def test_unparsable_bytes_degrade_to_none(self) -> None:
        assert outline_source(b"\x00\x01\x02") is None


class TestCaps:
    def test_symbol_cap_truncates_with_flag(self) -> None:
        source = "".join(
            f"def f{i}():\n    pass\n" for i in range(MAX_OUTLINE_SYMBOLS + 10)
        )
        outline = outline_source(source)
        assert outline is not None
        assert outline.truncated is True
        assert len(outline.symbols) == MAX_OUTLINE_SYMBOLS
        assert outline.symbols[0].name == "f0"
        assert outline.symbols[-1].name == f"f{MAX_OUTLINE_SYMBOLS - 1}"

    def test_render_byte_cap_truncates_within_budget(self) -> None:
        # Long names: few symbols, but the rendered text exceeds the byte cap.
        name = "n" * 200
        source = "".join(f"{name}{i} = lambda: None\n" for i in range(200))
        outline = outline_source(source)
        assert outline is not None
        assert outline.truncated is True
        assert len(outline.render().encode("utf-8")) <= MAX_OUTLINE_RENDER_BYTES

    def test_render_byte_cap_never_exceeded_when_not_truncated(self) -> None:
        source = "def f():\n    pass\n"
        outline = outline_source(source)
        assert outline is not None
        assert outline.truncated is False
        assert len(outline.render().encode("utf-8")) <= MAX_OUTLINE_RENDER_BYTES

    def test_depth_cap_truncates(self) -> None:
        source = "".join(
            f"{'    ' * depth}def f{depth}():\n{'    ' * (depth + 1)}pass\n"
            for depth in range(MAX_OUTLINE_DEPTH + 2)
        )
        outline = outline_source(source)
        assert outline is not None
        assert outline.truncated is True
        names = [s.name for s in outline.symbols]
        # f_k is collected at nesting depth k; the cap cuts below f64's scope.
        assert names[0] == "f0"
        assert len(names) == MAX_OUTLINE_DEPTH + 1
        assert names[-1] == ".".join(f"f{i}" for i in range(MAX_OUTLINE_DEPTH + 1))
        assert all("f65" not in name for name in names)


class TestOutlineFile:
    def test_outline_file_matches_outline_source(self, tmp_path: Path) -> None:
        file_path = tmp_path / "mod.py"
        file_path.write_text(SYMBOLS_SOURCE, encoding="utf-8")
        from_file = outline_file(file_path)
        from_source = outline_source(SYMBOLS_SOURCE, path=str(file_path))
        assert from_file is not None
        assert from_source is not None
        assert from_file == from_source

    def test_outline_file_accepts_str_path(self, tmp_path: Path) -> None:
        file_path = tmp_path / "mod.py"
        file_path.write_text("def f():\n    pass\n", encoding="utf-8")
        outline = outline_file(str(file_path))
        assert outline is not None
        assert [s.name for s in outline.symbols] == ["f"]

    def test_non_python_file_degrades(self, tmp_path: Path) -> None:
        file_path = tmp_path / "notes.txt"
        file_path.write_text("def f():\n    pass\n", encoding="utf-8")
        assert outline_file(file_path) is None

    def test_missing_file_degrades(self, tmp_path: Path) -> None:
        assert outline_file(tmp_path / "absent.py") is None

    def test_oversized_file_degrades(self, tmp_path: Path) -> None:
        file_path = tmp_path / "big.py"
        file_path.write_bytes(b"x = 1\n" * (MAX_OUTLINE_SOURCE_BYTES // 6 + 1))
        assert outline_file(file_path) is None


class TestRender:
    def test_render_format(self) -> None:
        outline = outline_source(NESTED_SOURCE)
        assert outline is not None
        assert outline.render().splitlines() == [
            "class Outer (lines 1-4)",
            "class Outer.Inner (lines 2-4)",
            "function Outer.Inner.method (lines 3-4)",
        ]

    def test_render_truncation_note(self) -> None:
        source = "".join(
            f"def f{i}():\n    pass\n" for i in range(MAX_OUTLINE_SYMBOLS + 1)
        )
        outline = outline_source(source)
        assert outline is not None
        rendered = outline.render()
        assert rendered.rstrip().endswith("bytes)")
        assert "truncated" in rendered.splitlines()[-1]
        assert len(rendered.encode("utf-8")) <= MAX_OUTLINE_RENDER_BYTES

    def test_symbol_schema_fields(self) -> None:
        symbol = OutlineSymbol(
            name="A.b", kind=SYMBOL_KIND_FUNCTION, line=3, end_line=7
        )
        assert symbol.name == "A.b"
        assert symbol.kind == "function"
        assert symbol.line == 3
        assert symbol.end_line == 7

    def test_outline_model_defaults(self) -> None:
        outline = FileOutline()
        assert outline.symbols == []
        assert outline.truncated is False
        assert outline.line_count == 0
