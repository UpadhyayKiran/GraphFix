"""
Purpose:
    Builds separate file and callable BM25 documents.

Input:
    Parsed repository records and source bounds.

Output:
    Deterministic SearchDocument tuples.

Pipeline stage:
    BM25 document preparation.
"""

from __future__ import annotations

from graphfix.parsing import ParsedRepository
from graphfix.retrieval.models import SearchDocument


def build_file_documents(
    parsed: ParsedRepository, *, max_source_chars: int = 12_000
) -> tuple[SearchDocument, ...]:
    """Represent each parsed Python file with path, imports, symbols, and bounded source."""
    if max_source_chars <= 0:
        raise ValueError("max_source_chars must be positive")
    documents: list[SearchDocument] = []
    for parsed_file in parsed.files:
        symbol_names = " ".join(symbol.qualified_name for symbol in parsed_file.symbols)
        imports = " ".join(
            f"{item.module} {' '.join(item.names)}" for item in parsed_file.imports
        )
        text = "\n".join(
            (
                f"file {parsed_file.file_path}",
                f"module {parsed_file.module}",
                f"symbols {symbol_names}",
                f"imports {imports}",
                parsed_file.source[:max_source_chars],
            )
        )
        documents.append(
            SearchDocument(
                document_id=f"file:{parsed_file.file_path}",
                kind="file",
                text=text,
                file_path=parsed_file.file_path,
                qualified_name=parsed_file.module,
            )
        )
    return tuple(documents)


def build_function_documents(
    parsed: ParsedRepository, *, max_source_chars: int = 8_000
) -> tuple[SearchDocument, ...]:
    """Represent functions and methods with identity, signature, docs, calls, and source."""
    if max_source_chars <= 0:
        raise ValueError("max_source_chars must be positive")
    documents: list[SearchDocument] = []
    for symbol in parsed.symbols:
        if symbol.kind not in {"function", "method"}:
            continue
        text = "\n".join(
            (
                f"{symbol.kind} {symbol.name}",
                f"qualified {symbol.qualified_name}",
                f"file {symbol.file_path}",
                f"parent {symbol.parent or ''}",
                f"signature {symbol.signature}",
                f"docstring {symbol.docstring}",
                f"calls {' '.join(symbol.calls)}",
                symbol.source[:max_source_chars],
            )
        )
        documents.append(
            SearchDocument(
                document_id=symbol.symbol_id,
                kind=symbol.kind,
                text=text,
                file_path=symbol.file_path,
                symbol_name=symbol.name,
                qualified_name=symbol.qualified_name,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
            )
        )
    return tuple(documents)
