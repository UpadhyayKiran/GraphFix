"""
Purpose:
    Rank existing GraphFix search documents by dense semantic similarity.

Input:
    Issue text, existing SearchDocument objects, and one embedding encoder.

Output:
    RankedCandidate objects ordered by cosine similarity.

Pipeline stage:
    Gold-free dense retrieval after document construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from graphfix.retrieval.models import RankedCandidate, SearchDocument

DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EmbeddingMatrix = tuple[tuple[float, ...], ...]


class TextEncoder(Protocol):
    """Small interface shared by the real encoder and deterministic test encoders."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one numeric embedding for every input text."""
        ...


class SentenceTransformerEncoder:
    """Optional local SentenceTransformers adapter loaded only when requested."""

    def __init__(self, model_name: str = DEFAULT_DENSE_MODEL) -> None:
        """Load one configured local/open-source sentence embedding model."""
        try:
            from sentence_transformers import (
                SentenceTransformer,  # type: ignore[import-not-found,unused-ignore]
            )
        except ImportError as error:
            raise RuntimeError(
                "Dense retrieval requires the optional 'dense' dependencies. "
                "Install with: py -3.11 -m pip install -e '.[dense]'"
            ) from error
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Encode text in batches and return ordinary lists for a small stable boundary."""
        embeddings = self._model.encode(
            list(texts), batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        return cast(list[list[float]], embeddings.tolist())


def document_fingerprint(documents: Sequence[SearchDocument]) -> str:
    """Return a stable SHA-256 digest of every ordered document ID and its actual text."""
    digest = hashlib.sha256()
    for document in documents:
        for value in (document.document_id, document.text):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


class EmbeddingCache:
    """Small filesystem and in-process cache for document embeddings only."""

    def __init__(self, root: Path) -> None:
        """Store the cache root; directories are created only when an entry is written."""
        self.root = root
        self._memory: dict[str, EmbeddingMatrix] = {}

    def _identity(
        self,
        *,
        repository: str,
        base_commit: str,
        model_name: str,
        document_level: str,
        documents: Sequence[SearchDocument],
    ) -> tuple[str, dict[str, object]]:
        """Build readable metadata and a filesystem-safe digest from all cache dimensions."""
        fingerprint = document_fingerprint(documents)
        metadata: dict[str, object] = {
            "repository": repository,
            "base_commit": base_commit.lower(),
            "model": model_name,
            "document_level": document_level,
            "document_count": len(documents),
            "document_fingerprint": fingerprint,
        }
        serialized = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), metadata

    def load_or_encode(
        self,
        documents: Sequence[SearchDocument],
        encoder: TextEncoder,
        *,
        repository: str,
        base_commit: str,
        model_name: str,
        document_level: str,
    ) -> EmbeddingMatrix:
        """
        Load exact document embeddings or encode and atomically replace a missing/bad entry.

        The key covers repository, exact commit, model, level, and SHA-256 content
        fingerprint. A corrupt or mismatched file is ignored and rebuilt. Query
        embeddings, rankings, evaluation, and gold data are never cached here.
        """
        import numpy as np  # type: ignore[import-untyped,unused-ignore]

        key, expected = self._identity(
            repository=repository,
            base_commit=base_commit,
            model_name=model_name,
            document_level=document_level,
            documents=documents,
        )
        if not documents:
            return ()
        if key in self._memory:
            return self._memory[key]
        path = self.root / f"{key}.npz"
        try:
            with np.load(path, allow_pickle=False) as stored:
                metadata = json.loads(str(stored["metadata"].item()))
                array = stored["embeddings"]
            metadata_matches = all(metadata.get(name) == value for name, value in expected.items())
            if (
                not metadata_matches
                or array.ndim != 2
                or array.shape[0] != len(documents)
                or metadata.get("embedding_shape") != list(array.shape)
            ):
                raise ValueError("Embedding cache metadata or shape does not match")
            embeddings = tuple(tuple(map(float, row)) for row in array)
            self._memory[key] = embeddings
            return embeddings
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

        encoded = encoder.encode([document.text for document in documents])
        if len(encoded) != len(documents):
            raise ValueError("Encoder must return one embedding per document")
        array = np.asarray(encoded, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("Document embeddings must form a two-dimensional matrix")
        metadata = {**expected, "embedding_shape": list(array.shape)}
        # Shape is useful for inspection but is excluded from key generation.
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / f".{key}.{os.getpid()}.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, embeddings=array, metadata=json.dumps(metadata))
        os.replace(temporary, path)
        embeddings = tuple(tuple(map(float, row)) for row in array)
        self._memory[key] = embeddings
        return embeddings


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute dot(left, right) / (norm(left) * norm(right)), or zero for a zero vector."""
    if len(left) != len(right):
        raise ValueError("Embedding dimensions must match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


class DenseIndex:
    """Dense index over the same ordered SearchDocument objects used by BM25."""

    def __init__(
        self,
        documents: Sequence[SearchDocument],
        encoder: TextEncoder,
        *,
        document_embeddings: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Encode all documents once; no benchmark gold is accepted or stored."""
        self.documents = tuple(documents)
        self.encoder = encoder
        encoded = (
            document_embeddings
            if document_embeddings is not None
            else encoder.encode([document.text for document in self.documents])
        )
        if len(encoded) != len(self.documents):
            raise ValueError("Encoder must return one embedding per document")
        self._document_embeddings = tuple(tuple(map(float, vector)) for vector in encoded)

    def search(self, query: str, *, top_k: int = 10) -> tuple[RankedCandidate, ...]:
        """Rank documents by similarity(q,d) = cosine(embedding(q), embedding(d))."""
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0 or not self.documents:
            return ()
        query_vectors = self.encoder.encode([query])
        if len(query_vectors) != 1:
            raise ValueError("Encoder must return one embedding for the query")
        query_embedding = tuple(map(float, query_vectors[0]))
        scored = [
            (cosine_similarity(query_embedding, embedding), document)
            for document, embedding in zip(
                self.documents, self._document_embeddings, strict=True
            )
        ]
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
