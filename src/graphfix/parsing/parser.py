"""
Purpose:
    Parses Python repositories without importing target code.

Input:
    A verified historical worktree.

Output:
    Files, imports, symbols, calls, inheritance, and failures.

Pipeline stage:
    Static extraction before graph and retrieval.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from graphfix.parsing.models import (
    ImportRecord,
    ParsedFile,
    ParsedRepository,
    ParserStats,
    SourceSymbol,
    SymbolKind,
)

_IGNORED_PARTS: Final[frozenset[str]] = frozenset(
    {".git", ".hg", ".svn", ".venv", "venv", "__pycache__", "node_modules"}
)


def _module_name(path: str) -> str:
    """
    Purpose:
        Convert a repository-relative Python path into its import-style module name.

    Args:
        path: The file or directory path to use.

    Returns:
        A `str` result produced by the operation.

    Pipeline role:
        Python AST parsing after checkout verification.
    """
    without_suffix = path[:-3] if path.endswith(".py") else path
    parts = without_suffix.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _expression_name(node: ast.AST) -> str | None:
    """
    Purpose:
        Convert a simple AST name or attribute expression into readable dotted text.

    Args:
        node: The AST or graph node being inspected.

    Returns:
        A `str | None` result produced by the operation.

    Pipeline role:
        Python AST parsing after checkout verification.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        return _expression_name(node.value)
    return None


def _source_segment(source_lines: tuple[bytes, ...], node: ast.AST) -> str | None:
    """Extract one AST segment without repeatedly splitting the whole source file."""
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    col_offset = getattr(node, "col_offset", None)
    end_col_offset = getattr(node, "end_col_offset", None)
    if not all(
        isinstance(value, int)
        for value in (lineno, end_lineno, col_offset, end_col_offset)
    ):
        return None
    assert isinstance(lineno, int)
    assert isinstance(end_lineno, int)
    assert isinstance(col_offset, int)
    assert isinstance(end_col_offset, int)
    start_line = lineno - 1
    end_line = end_lineno - 1
    if start_line == end_line:
        payload = source_lines[start_line][col_offset:end_col_offset]
    else:
        payload = b"".join(
            (
                source_lines[start_line][col_offset:],
                *source_lines[start_line + 1 : end_line],
                source_lines[end_line][:end_col_offset],
            )
        )
    return payload.decode("utf-8")


def _signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef, source_lines: tuple[bytes, ...]
) -> str:
    """
    Purpose:
        Extract the exact source signature from `def` through the closing parenthesis.

    Args:
        node: The AST or graph node being inspected.
        source_lines: The source lines value used by this operation.

    Returns:
        A `str` result produced by the operation.

    Pipeline role:
        Python AST parsing after checkout verification.
    """
    segment = _source_segment(source_lines, node)
    if segment:
        return segment.splitlines()[0].strip().removesuffix(":")
    return f"{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}def {node.name}(...)"


class _SymbolVisitor(ast.NodeVisitor):
    """
    Purpose:
        Walk one Python AST and record imports, classes, functions, methods,
        inheritance, and calls.

    Pipeline role:
        Python AST parsing after checkout verification.
    """
    def __init__(self, *, file_path: str, module: str, source: str) -> None:
        """
        Purpose:
            Initialize the object and store the configuration needed by later operations.

        Args:
            file_path: Repository-relative file path.
            module: Python module name.
            source: Python source text.

        Returns:
            Nothing. The operation updates state, writes output, or performs a check directly.

        Pipeline role:
            Python AST parsing after checkout verification.
        """
        self.file_path = file_path
        self.module = module
        self.source = source
        self.source_lines = tuple(
            line.encode("utf-8") for line in source.splitlines(keepends=True)
        )
        self.parents: list[tuple[str, SymbolKind]] = []
        self.symbols: list[SourceSymbol] = []
        self.imports: list[ImportRecord] = []

    def visit_Import(self, node: ast.Import) -> None:
        """
        Purpose:
            Visit an AST Import node and record the structure GraphFix needs.

        Args:
            node: The AST or graph node being inspected.

        Returns:
            Nothing. The operation updates state, writes output, or performs a check directly.

        Pipeline role:
            Python AST parsing after checkout verification.
        """
        for alias in node.names:
            self.imports.append(ImportRecord(module=alias.name, line=node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        Purpose:
            Visit an AST ImportFrom node and record the structure GraphFix needs.

        Args:
            node: The AST or graph node being inspected.

        Returns:
            Nothing. The operation updates state, writes output, or performs a check directly.

        Pipeline role:
            Python AST parsing after checkout verification.
        """
        prefix = "." * node.level
        module = f"{prefix}{node.module or ''}"
        self.imports.append(
            ImportRecord(
                module=module,
                names=tuple(alias.name for alias in node.names),
                line=node.lineno,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Purpose:
            Visit an AST ClassDef node and record the structure GraphFix needs.

        Args:
            node: The AST or graph node being inspected.

        Returns:
            Nothing. The operation updates state, writes output, or performs a check directly.

        Pipeline role:
            Python AST parsing after checkout verification.
        """
        parent = self.parents[-1][0] if self.parents else None
        qualified = ".".join(filter(None, (self.module, parent, node.name)))
        calls = self._calls(node, skip_nested=True)
        self.symbols.append(
            SourceSymbol(
                symbol_id=f"class:{self.file_path}:{qualified}:{node.lineno}",
                name=node.name,
                qualified_name=qualified,
                kind="class",
                file_path=self.file_path,
                module=self.module,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                parent=parent,
                signature=f"class {node.name}",
                docstring=ast.get_docstring(node, clean=False) or "",
                source=_source_segment(self.source_lines, node) or "",
                bases=tuple(filter(None, (_expression_name(base) for base in node.bases))),
                calls=calls,
            )
        )
        self.parents.append((node.name, "class"))
        self.generic_visit(node)
        self.parents.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Purpose:
            Visit an AST FunctionDef node and record the structure GraphFix needs.

        Args:
            node: The AST or graph node being inspected.

        Returns:
            Nothing. The operation updates state, writes output, or performs a check directly.

        Pipeline role:
            Python AST parsing after checkout verification.
        """
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """
        Purpose:
            Visit an AST AsyncFunctionDef node and record the structure GraphFix needs.

        Args:
            node: The AST or graph node being inspected.

        Returns:
            Nothing. The operation updates state, writes output, or performs a check directly.

        Pipeline role:
            Python AST parsing after checkout verification.
        """
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """
        Purpose:
            Extract one function or method, including its nesting, signature,
            docstring, line span, source, and calls.

        Args:
            node: The AST or graph node being inspected.

        Returns:
            Nothing. The operation updates state, writes output, or performs a check directly.

        Pipeline role:
            Python AST parsing after checkout verification.
        """
        parent_class = next(
            (name for name, kind in reversed(self.parents) if kind == "class"), None
        )
        immediate_parent_kind = self.parents[-1][1] if self.parents else None
        kind: SymbolKind = "method" if immediate_parent_kind == "class" else "function"
        lexical_parents = tuple(name for name, _kind in self.parents)
        qualified = ".".join((self.module, *lexical_parents, node.name))
        self.symbols.append(
            SourceSymbol(
                symbol_id=f"{kind}:{self.file_path}:{qualified}:{node.lineno}",
                name=node.name,
                qualified_name=qualified,
                kind=kind,
                file_path=self.file_path,
                module=self.module,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                parent=parent_class,
                signature=_signature(node, self.source_lines),
                docstring=ast.get_docstring(node, clean=False) or "",
                source=_source_segment(self.source_lines, node) or "",
                calls=self._calls(node, skip_nested=True),
            )
        )
        self.parents.append((node.name, kind))
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(child)
        self.parents.pop()

    @staticmethod
    def _calls(node: ast.AST, *, skip_nested: bool) -> tuple[str, ...]:
        """
        Purpose:
            Collect static call names inside one function body without entering
            nested definitions.

        Args:
            node: The AST or graph node being inspected.
            skip_nested: The skip nested value used by this operation.

        Returns:
            A `tuple[str, ...]` result produced by the operation.

        Pipeline role:
            Python AST parsing after checkout verification.
        """
        calls: set[str] = set()

        class CallVisitor(ast.NodeVisitor):
            """
            Purpose:
                Collects calls from one function body while skipping nested definitions.

            Pipeline role:
                Python AST parsing after checkout verification.
            """
            def visit_Call(self, call: ast.Call) -> None:
                """
                Purpose:
                    Visit an AST Call node and record the structure GraphFix needs.

                Args:
                    call: The call value used by this operation.

                Returns:
                    Nothing. The visitor records calls directly.

                Pipeline role:
                    Python AST parsing after checkout verification.
                """
                name = _expression_name(call.func)
                if name:
                    calls.add(name)
                self.generic_visit(call)

            def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
                """
                Purpose:
                    Visit an AST FunctionDef node and record the structure GraphFix needs.

                Args:
                    child: The child value used by this operation.

                Returns:
                    Nothing. The visitor deliberately skips this nested function.

                Pipeline role:
                    Python AST parsing after checkout verification.
                """
                if child is node or not skip_nested:
                    self.generic_visit(child)

            def visit_AsyncFunctionDef(self, child: ast.AsyncFunctionDef) -> None:
                """
                Purpose:
                    Visit an AST AsyncFunctionDef node and record the structure GraphFix needs.

                Args:
                    child: The child value used by this operation.

                Returns:
                    Nothing. The visitor deliberately skips this nested function.

                Pipeline role:
                    Python AST parsing after checkout verification.
                """
                if child is node or not skip_nested:
                    self.generic_visit(child)

            def visit_ClassDef(self, child: ast.ClassDef) -> None:
                """
                Purpose:
                    Visit an AST ClassDef node and record the structure GraphFix needs.

                Args:
                    child: The child value used by this operation.

                Returns:
                    Nothing. The visitor deliberately skips this nested class.

                Pipeline role:
                    Python AST parsing after checkout verification.
                """
                if child is node or not skip_nested:
                    self.generic_visit(child)

        CallVisitor().visit(node)
        return tuple(sorted(calls))


class PythonRepositoryParser:
    """Parse all safe regular Python files under a repository root."""

    def __init__(self, *, max_source_chars: int = 20_000) -> None:
        """
        Purpose:
            Initialize the object and store the configuration needed by later operations.

        Args:
            max_source_chars: The max source chars value used by this operation.

        Returns:
            Nothing. The operation updates state, writes output, or performs a check directly.

        Raises:
            ValueError: If the input or operation cannot be completed safely.

        Pipeline role:
            Python AST parsing after checkout verification.
        """
        if max_source_chars <= 0:
            raise ValueError("max_source_chars must be positive")
        self._max_source_chars = max_source_chars

    def parse(self, root: Path) -> ParsedRepository:
        """Return deterministic parser output while retaining per-file failures."""
        resolved_root = root.resolve()
        if not resolved_root.is_dir():
            raise ValueError(f"Repository root is not a directory: {resolved_root}")
        scanned = sorted(
            (
                path for path in resolved_root.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and not any(
                    part in _IGNORED_PARTS
                    for part in path.relative_to(resolved_root).parts
                )
            ),
            key=lambda path: path.relative_to(resolved_root).as_posix(),
        )
        candidates = [path for path in scanned if path.suffix == ".py"]
        files: list[ParsedFile] = []
        failures: list[str] = []
        for path in candidates:
            relative = path.relative_to(resolved_root).as_posix()
            try:
                source = path.read_text(encoding="utf-8-sig")
                tree = ast.parse(source, filename=relative, type_comments=True)
                visitor = _SymbolVisitor(
                    file_path=relative,
                    module=_module_name(relative),
                    source=source,
                )
                visitor.visit(tree)
                bounded_source = source[: self._max_source_chars]
                files.append(
                    ParsedFile(
                        file_path=relative,
                        module=_module_name(relative),
                        source=bounded_source,
                        imports=tuple(visitor.imports),
                        symbols=tuple(visitor.symbols),
                    )
                )
            except (OSError, RecursionError, SyntaxError, UnicodeError, ValueError) as error:
                failures.append(f"{relative}: {type(error).__name__}: {error}")

        symbols = [symbol for file in files for symbol in file.symbols]
        stats = ParserStats(
            files_scanned=len(scanned),
            python_files_parsed=len(files),
            parse_failures=len(failures),
            classes=sum(symbol.kind == "class" for symbol in symbols),
            functions=sum(symbol.kind == "function" for symbol in symbols),
            methods=sum(symbol.kind == "method" for symbol in symbols),
            imports=sum(len(file.imports) for file in files),
            calls=sum(len(symbol.calls) for symbol in symbols),
        )
        return ParsedRepository(
            root=str(resolved_root),
            files=tuple(files),
            failures=tuple(failures),
            stats=stats,
        )
