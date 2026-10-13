"""
Purpose:
    Verify parsing behavior without requiring network access.

Input:
    Small temporary repositories, records, models, or fixtures created by the tests.

Output:
    Passing assertions when the implementation follows the Review-1 contract.

Pipeline stage:
    Regression testing for the GraphFix implementation.
"""

import ast
from pathlib import Path
from unittest.mock import patch

from graphfix.parsing import PythonRepositoryParser
from graphfix.parsing.parser import _source_segment


def test_extracts_classes_functions_methods_imports_inheritance_and_calls(tmp_path: Path) -> None:
    """Verify extracts classes functions methods imports inheritance and calls."""
    (tmp_path / "base.py").write_text("class Base:\n    pass\n", encoding="utf-8")
    (tmp_path / "service.py").write_text(
        '''"""Fixture module."""
import json
from base import Base

def helper(value: str) -> str:
    """Return a normalized value."""
    return value.strip()

class Service(Base):
    def run(self, value: str) -> str:
        return helper(value)
''',
        encoding="utf-8",
    )

    parsed = PythonRepositoryParser().parse(tmp_path)

    assert parsed.stats.files_scanned == 2
    assert parsed.stats.python_files_parsed == 2
    assert parsed.stats.parse_failures == 0
    assert parsed.stats.classes == 2
    assert parsed.stats.functions == 1
    assert parsed.stats.methods == 1
    service = next(file for file in parsed.files if file.file_path == "service.py")
    assert [(item.module, item.names) for item in service.imports] == [
        ("json", ()),
        ("base", ("Base",)),
    ]
    derived = next(symbol for symbol in service.symbols if symbol.name == "Service")
    method = next(symbol for symbol in service.symbols if symbol.name == "run")
    assert derived.bases == ("Base",)
    assert method.kind == "method"
    assert method.parent == "Service"
    assert "helper" in method.calls
    assert method.start_line < method.end_line
    assert "def run" in method.signature


def test_parse_failure_does_not_abort_other_files(tmp_path: Path) -> None:
    """Verify parse failure does not abort other files."""
    (tmp_path / "good.py").write_text("def valid():\n    return 1\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("def invalid(:\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("fixture", encoding="utf-8")

    parsed = PythonRepositoryParser().parse(tmp_path)

    assert parsed.stats.files_scanned == 3
    assert parsed.stats.python_files_parsed == 1
    assert parsed.stats.parse_failures == 1
    assert parsed.files[0].file_path == "good.py"
    assert parsed.failures[0].startswith("bad.py: SyntaxError:")


def test_recursive_ast_failure_is_recorded_without_aborting_repository(tmp_path: Path) -> None:
    """Verify recursive ast failure is recorded without aborting repository."""
    (tmp_path / "deep.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "good.py").write_text("def valid():\n    return 1\n", encoding="utf-8")

    original_parse = __import__("ast").parse

    def parse_with_one_failure(source: str, **kwargs: object) -> object:
        """Perform parse with one failure."""
        if kwargs.get("filename") == "deep.py":
            raise RecursionError("fixture nesting is too deep")
        return original_parse(source, **kwargs)

    with patch("graphfix.parsing.parser.ast.parse", side_effect=parse_with_one_failure):
        parsed = PythonRepositoryParser().parse(tmp_path)

    assert parsed.stats.python_files_parsed == 1
    assert parsed.stats.parse_failures == 1
    assert parsed.failures == ("deep.py: RecursionError: fixture nesting is too deep",)


def test_cached_source_segment_matches_ast_for_unicode_and_multiline_source() -> None:
    """Verify cached source segment matches ast for unicode and multiline source."""
    source = 'def example(value: str = "café"):\n    return (\n        value.strip()\n    )\n'
    tree = ast.parse(source)
    function = tree.body[0]
    lines = tuple(line.encode("utf-8") for line in source.splitlines(keepends=True))

    assert _source_segment(lines, function) == ast.get_source_segment(source, function)


def test_nested_functions_have_unique_qualified_names_and_are_not_methods(tmp_path: Path) -> None:
    """Verify nested functions have unique qualified names and are not methods."""
    (tmp_path / "nested.py").write_text(
        "class Service:\n"
        "    def first(self):\n"
        "        def helper():\n            return 1\n"
        "        return helper()\n"
        "    def second(self):\n"
        "        def helper():\n            return 2\n"
        "        return helper()\n",
        encoding="utf-8",
    )

    parsed = PythonRepositoryParser().parse(tmp_path)
    helpers = [symbol for symbol in parsed.symbols if symbol.name == "helper"]

    assert [symbol.kind for symbol in helpers] == ["function", "function"]
    assert len({symbol.symbol_id for symbol in parsed.symbols}) == len(parsed.symbols)
    assert {symbol.qualified_name for symbol in helpers} == {
        "nested.Service.first.helper",
        "nested.Service.second.helper",
    }


def test_redefined_symbol_names_keep_distinct_line_based_identities(tmp_path: Path) -> None:
    """Verify redefined symbol names keep distinct line based identities."""
    (tmp_path / "overload.py").write_text(
        "def choose(value: int):\n    return value\n\n"
        "def choose(value: str):\n    return value\n",
        encoding="utf-8",
    )

    parsed = PythonRepositoryParser().parse(tmp_path)

    assert [symbol.qualified_name for symbol in parsed.symbols] == [
        "overload.choose",
        "overload.choose",
    ]
    assert len({symbol.symbol_id for symbol in parsed.symbols}) == 2


def test_signature_has_parentheses_and_module_control_flow_is_parsed(tmp_path: Path) -> None:
    """Verify signature has parentheses and module control flow is parsed."""
    (tmp_path / "conditional.py").write_text(
        "if enabled:\n    async def endpoint(request):\n        return request\n",
        encoding="utf-8",
    )

    parsed = PythonRepositoryParser().parse(tmp_path)
    endpoint = next(symbol for symbol in parsed.symbols if symbol.name == "endpoint")

    assert endpoint.parent is None
    assert endpoint.signature == "async def endpoint(request)"
