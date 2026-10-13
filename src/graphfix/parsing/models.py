"""
Purpose:
    Defines typed parser records and statistics.

Input:
    AST-derived file and symbol information.

Output:
    Serializable parsed-repository records.

Pipeline stage:
    Static parsing output.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SymbolKind = Literal["class", "function", "method"]


class ParsingModel(BaseModel):
    """Strict immutable base for parser output."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ImportRecord(ParsingModel):
    """One import statement normalized to its referenced module names."""

    module: str
    names: tuple[str, ...] = ()
    line: int = Field(ge=1)


class SourceSymbol(ParsingModel):
    """A class, function, or method extracted from a Python source file."""

    symbol_id: str
    name: str
    qualified_name: str
    kind: SymbolKind
    file_path: str
    module: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    parent: str | None = None
    signature: str
    docstring: str = ""
    source: str = ""
    bases: tuple[str, ...] = ()
    calls: tuple[str, ...] = ()


class ParsedFile(ParsingModel):
    """Normalized content extracted from one successfully parsed Python file."""

    file_path: str
    module: str
    source: str
    imports: tuple[ImportRecord, ...]
    symbols: tuple[SourceSymbol, ...]


class ParserStats(ParsingModel):
    """Repository-wide parser counts suitable for reports and demos."""

    files_scanned: int = Field(ge=0)
    python_files_parsed: int = Field(ge=0)
    parse_failures: int = Field(ge=0)
    classes: int = Field(ge=0)
    functions: int = Field(ge=0)
    methods: int = Field(ge=0)
    imports: int = Field(ge=0)
    calls: int = Field(ge=0)


class ParsedRepository(ParsingModel):
    """All parser output for one repository root."""

    root: str
    files: tuple[ParsedFile, ...]
    failures: tuple[str, ...]
    stats: ParserStats

    @property
    def symbols(self) -> tuple[SourceSymbol, ...]:
        """Return symbols in deterministic file/source order."""
        return tuple(symbol for file in self.files for symbol in file.symbols)
