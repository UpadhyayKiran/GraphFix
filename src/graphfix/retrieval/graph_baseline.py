"""
Purpose:
    Add deterministic RepositoryGraph neighborhood evidence to a retrieval ranking.

Input:
    A gold-free retrieval ranking, eligible documents, and actual graph nodes/edges.

Output:
    A graph-derived ranking and an RRF-fused graph-augmented ranking.

Pipeline stage:
    Static graph-informed localization after hybrid retrieval.
"""

from __future__ import annotations

from collections.abc import Sequence

from graphfix.graph import GraphNode, RepositoryGraph
from graphfix.graph.models import EdgeType
from graphfix.retrieval.hybrid import reciprocal_rank_fusion
from graphfix.retrieval.models import RankedCandidate, SearchDocument

DEFAULT_GRAPH_EDGES: tuple[EdgeType, ...] = ("CALLS", "IMPORTS", "INHERITS")


def _project_node(node: GraphNode, eligible: dict[str, SearchDocument]) -> str | None:
    """Project a reached graph node onto an eligible file or callable document."""
    if node.node_id in eligible:
        return node.node_id
    file_id = f"file:{node.file_path}" if node.file_path else None
    if file_id in eligible:
        return file_id
    return None


def graph_candidate_ranking(
    retrieval_ranking: Sequence[RankedCandidate],
    documents: Sequence[SearchDocument],
    graph: RepositoryGraph,
    *,
    seed_top_k: int = 10,
    depth: int = 1,
    edge_types: Sequence[EdgeType] = DEFAULT_GRAPH_EDGES,
) -> tuple[RankedCandidate, ...]:
    """
    Rank candidates reached from top retrieval seeds through real graph edges.

    Seeds are the first `seed_top_k` retrieval candidates that resolve to graph
    nodes. Breadth-first traversal follows only selected existing edges, in either
    direction, up to `depth`. Reached nodes are projected to an eligible document:
    a callable stays that callable, while any node with a file path can support its
    file document. Candidates are ordered by minimum hop distance, originating seed
    rank, then document ID. No learned weights or gold locations are used.
    """
    if seed_top_k <= 0:
        raise ValueError("seed_top_k must be positive")
    if depth <= 0:
        raise ValueError("depth must be positive")
    eligible = {document.document_id: document for document in documents}
    allowed_edges = set(edge_types)
    evidence_order: dict[str, tuple[int, int, str]] = {}

    for seed in retrieval_ranking[:seed_top_k]:
        if graph.get_node(seed.document_id) is None:
            continue
        frontier = {seed.document_id}
        visited = {seed.document_id}
        for distance in range(1, depth + 1):
            reached: set[str] = set()
            for edge in graph.edges:
                if edge.edge_type not in allowed_edges:
                    continue
                if edge.source in frontier:
                    reached.add(edge.target)
                if edge.target in frontier:
                    reached.add(edge.source)
            reached -= visited
            for node_id in sorted(reached):
                node = graph.get_node(node_id)
                if node is None:
                    continue
                document_id = _project_node(node, eligible)
                if document_id is not None:
                    order = (distance, seed.rank, document_id)
                    evidence_order[document_id] = min(
                        evidence_order.get(document_id, order), order
                    )
            visited.update(reached)
            frontier = reached
            if not frontier:
                break

    ordered_ids = sorted(evidence_order, key=evidence_order.__getitem__)
    return tuple(
        RankedCandidate(
            rank=rank,
            score=1.0 / order[0],
            document_id=document.document_id,
            kind=document.kind,
            file_path=document.file_path,
            symbol_name=document.symbol_name,
            qualified_name=document.qualified_name,
            start_line=document.start_line,
            end_line=document.end_line,
        )
        for rank, document_id in enumerate(ordered_ids, start=1)
        for order in (evidence_order[document_id],)
        for document in (eligible[document_id],)
    )


def graph_augmented_ranking(
    retrieval_ranking: Sequence[RankedCandidate],
    documents: Sequence[SearchDocument],
    graph: RepositoryGraph,
    *,
    top_k: int = 10,
    seed_top_k: int = 10,
    depth: int = 1,
    rrf_k: int = 60,
    edge_types: Sequence[EdgeType] = DEFAULT_GRAPH_EDGES,
) -> tuple[RankedCandidate, ...]:
    """Fuse the original ranking and deterministic graph-derived ranking with RRF."""
    graph_ranking = graph_candidate_ranking(
        retrieval_ranking,
        documents,
        graph,
        seed_top_k=seed_top_k,
        depth=depth,
        edge_types=edge_types,
    )
    return reciprocal_rank_fusion(
        (retrieval_ranking, graph_ranking), top_k=top_k, rrf_k=rrf_k
    )
