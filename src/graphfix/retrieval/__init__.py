"""
Purpose:
    Expose the public retrieval names used by other GraphFix modules.

Input:
    Imports from the implementation modules in this package.

Output:
    A small, stable package-level import surface.

Pipeline stage:
    Package wiring for the GraphFix Review-1 pipeline.
"""

from graphfix.retrieval.bm25 import BM25Index, tokenize
from graphfix.retrieval.dense import (
    DEFAULT_DENSE_MODEL,
    DenseIndex,
    EmbeddingCache,
    SentenceTransformerEncoder,
    TextEncoder,
    cosine_similarity,
    document_fingerprint,
)
from graphfix.retrieval.documents import build_file_documents, build_function_documents
from graphfix.retrieval.graph_baseline import (
    DEFAULT_GRAPH_EDGES,
    graph_augmented_ranking,
    graph_candidate_ranking,
)
from graphfix.retrieval.hybrid import reciprocal_rank_fusion
from graphfix.retrieval.models import RankedCandidate, SearchDocument

__all__ = [
    "BM25Index",
    "DEFAULT_DENSE_MODEL",
    "DenseIndex",
    "EmbeddingCache",
    "DEFAULT_GRAPH_EDGES",
    "RankedCandidate",
    "SearchDocument",
    "SentenceTransformerEncoder",
    "TextEncoder",
    "build_file_documents",
    "build_function_documents",
    "cosine_similarity",
    "document_fingerprint",
    "graph_augmented_ranking",
    "graph_candidate_ranking",
    "reciprocal_rank_fusion",
    "tokenize",
]
