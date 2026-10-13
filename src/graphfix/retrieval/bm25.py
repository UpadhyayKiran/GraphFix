"""
Purpose:
    Ranks lexical documents with explicit Okapi BM25.

Input:
    Documents, query, k1, b, and top-k.

Output:
    Deterministically ordered candidates.

Pipeline stage:
    BM25 sparse-retrieval baseline.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Final

from graphfix.retrieval.models import RankedCandidate, SearchDocument

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> tuple[str, ...]:
    """
    Purpose:
        Convert issue text or code text into lowercase search terms.

    Args:
        text: Natural language, a path, or source-code text.

    Returns:
        Tokens split at punctuation, path separators, underscores, and CamelCase boundaries.

    How it works:
        CamelCase, snake_case, and path separators are expanded before the remaining
        letters and numbers are collected. This lets issue words match code identifiers.

    Pipeline role:
        Shared code-aware tokenization for file and function BM25 retrieval.
    """
    expanded = _CAMEL_BOUNDARY.sub(" ", text).replace("_", " ").replace("/", " ")
    return tuple(token.lower() for token in _TOKEN.findall(expanded))


class BM25Index:
    """Immutable-in-practice BM25 index over ordered search documents."""

    def __init__(
        self,
        documents: Sequence[SearchDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """
        Purpose:
            Build the statistics needed to score the supplied search documents with Okapi BM25.

        Args:
            documents: Searchable documents used to build the BM25 index.
            k1: Positive term-frequency saturation setting. Larger values give repeated terms
                more influence before their benefit levels off.
            b: Document-length normalization setting from zero to one.

        Returns:
            Nothing. The operation updates state, writes output, or performs a check directly.

        Raises:
            ValueError: If k1 is not positive or b is outside zero to one.

        How it works:
            1. Tokenize every document and count each term in that document.
            2. Record document lengths and their average length.
            3. Count how many documents contain each term.
            4. Calculate inverse document frequency so rarer terms receive more weight.

        Pipeline role:
            Gold-free BM25 indexing and ranking.
        """
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.documents: Final[tuple[SearchDocument, ...]] = tuple(documents)
        self.k1: Final[float] = k1
        self.b: Final[float] = b
        self._tokens = tuple(tokenize(document.text) for document in self.documents)
        self._frequencies = tuple(Counter(tokens) for tokens in self._tokens)
        self._lengths = tuple(len(tokens) for tokens in self._tokens)
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        document_frequencies: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequencies.update(set(tokens))
        count = len(self.documents)
        self._idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def search(self, query: str, *, top_k: int = 10) -> tuple[RankedCandidate, ...]:
        """
        Purpose:
            Rank the indexed files or functions by lexical similarity to an issue statement.

        Args:
            query: Natural-language issue text.
            top_k: Maximum number of candidates to return.

        Returns:
            Ranked candidates ordered by decreasing BM25 score, then document ID for stable ties.

        Raises:
            ValueError: If top_k is negative.

        How it works:
            1. Tokenize the query with the same code-aware tokenizer used for documents.
            2. For each document, combine term frequency, inverse document frequency,
               average length, k1, and b using the Okapi BM25 formula.
            3. Sort scores and return at most top_k candidates.

        Pipeline role:
            Gold-free file or function ranking. BM25 does not use graph edges.
        """
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0 or not self.documents:
            return ()
        query_terms = tokenize(query)
        scored: list[tuple[float, SearchDocument]] = []
        for index, document in enumerate(self.documents):
            score = 0.0
            length = self._lengths[index]
            normalization = (
                1.0 - self.b + self.b * length / self._average_length
                if self._average_length
                else 1.0
            )
            for term in query_terms:
                frequency = self._frequencies[index].get(term, 0)
                if not frequency:
                    continue
                score += self._idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0)
                    / (frequency + self.k1 * normalization)
                )
            scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], item[1].document_id))
        return tuple(
            RankedCandidate(
                rank=rank,
                score=float(score),
                document_id=document.document_id,
                kind=document.kind,
                file_path=document.file_path,
                symbol_name=document.symbol_name,
                qualified_name=document.qualified_name,
                start_line=document.start_line,
                end_line=document.end_line,
            )
            for rank, (score, document) in enumerate(scored[:top_k], start=1)
        )
