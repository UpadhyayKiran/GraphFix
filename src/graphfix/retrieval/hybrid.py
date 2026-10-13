"""
Purpose:
    Fuse sparse BM25 and dense rankings with Reciprocal Rank Fusion (RRF).

Input:
    Two or more gold-free rankings containing the same candidate type.

Output:
    One deterministic hybrid ranking of RankedCandidate objects.

Pipeline stage:
    Gold-free hybrid retrieval after independent BM25 and dense retrieval.
"""

from __future__ import annotations

from collections.abc import Sequence

from graphfix.retrieval.models import RankedCandidate


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankedCandidate]],
    *,
    top_k: int = 10,
    rrf_k: int = 60,
) -> tuple[RankedCandidate, ...]:
    """
    Fuse candidate ranks using score(d) = sum(1 / (rrf_k + rank_i(d))).

    RRF uses ranks rather than raw scores, so BM25 and cosine similarity do not
    need arbitrary normalization onto the same numerical scale. Candidates may
    occur in only one input ranking. Equal fusion scores are ordered by stable
    document ID. This function accepts no gold information.
    """
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if top_k == 0:
        return ()

    scores: dict[str, float] = {}
    candidates: dict[str, RankedCandidate] = {}
    for ranking in rankings:
        for candidate in ranking:
            candidates.setdefault(candidate.document_id, candidate)
            scores[candidate.document_id] = scores.get(candidate.document_id, 0.0) + (
                1.0 / (rrf_k + candidate.rank)
            )

    ordered_ids = sorted(scores, key=lambda document_id: (-scores[document_id], document_id))
    fused: list[RankedCandidate] = []
    for rank, document_id in enumerate(ordered_ids[:top_k], start=1):
        candidate = candidates[document_id]
        fused.append(
            candidate.model_copy(update={"rank": rank, "score": scores[document_id]})
        )
    return tuple(fused)
