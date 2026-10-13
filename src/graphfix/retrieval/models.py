"""
Purpose:
    Defines BM25 documents and ranked candidates.

Input:
    Lexical text and source identity metadata.

Output:
    Typed retrieval input and output records.

Pipeline stage:
    Sparse retrieval contracts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocumentKind = Literal["file", "function", "method"]


class RetrievalModel(BaseModel):
    """
    Purpose:
        Base model that rejects unexpected retrieval fields.

    Pipeline role:
        Gold-free BM25 indexing and ranking.
    """
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SearchDocument(RetrievalModel):
    """
    Purpose:
        Stores one file or function document that BM25 can search.

    Contains:
        document_id: The document id value.
        kind: The kind value.
        text: The text value.
        file_path: The file path value.
        symbol_name: The symbol name value.
        qualified_name: The qualified name value.
        start_line: The start line value.
        end_line: The end line value.

    Pipeline role:
        Gold-free BM25 indexing and ranking.
    """
    document_id: str
    kind: DocumentKind
    text: str
    file_path: str
    symbol_name: str | None = None
    qualified_name: str | None = None
    start_line: int | None = None
    end_line: int | None = None


class RankedCandidate(RetrievalModel):
    """
    Purpose:
        Stores one BM25 result with its rank, score, path, and optional symbol details.

    Contains:
        rank: The rank value.
        score: The score value.
        document_id: The document id value.
        kind: The kind value.
        file_path: The file path value.
        symbol_name: The symbol name value.
        qualified_name: The qualified name value.
        start_line: The start line value.
        end_line: The end line value.

    Pipeline role:
        Gold-free BM25 indexing and ranking.
    """
    rank: int = Field(ge=1)
    score: float
    document_id: str
    kind: DocumentKind
    file_path: str
    symbol_name: str | None = None
    qualified_name: str | None = None
    start_line: int | None = None
    end_line: int | None = None
