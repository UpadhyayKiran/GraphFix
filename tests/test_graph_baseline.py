from graphfix.graph import GraphEdge, GraphNode, RepositoryGraph
from graphfix.retrieval import (
    RankedCandidate,
    SearchDocument,
    graph_augmented_ranking,
    graph_candidate_ranking,
)


def _file(name: str) -> SearchDocument:
    return SearchDocument(
        document_id=f"file:{name}.py", kind="file", text=name, file_path=f"{name}.py"
    )


def _ranked(document: SearchDocument, rank: int) -> RankedCandidate:
    return RankedCandidate(rank=rank, score=1.0, **document.model_dump(exclude={"text"}))


def _graph() -> RepositoryGraph:
    nodes = tuple(
        GraphNode(
            node_id=f"file:{name}.py", node_type="file", name=f"{name}.py", file_path=f"{name}.py"
        )
        for name in ("a", "b", "c", "unconnected")
    )
    return RepositoryGraph(
        nodes=nodes,
        edges=(
            GraphEdge(source="file:a.py", target="file:b.py", edge_type="IMPORTS"),
            GraphEdge(source="file:b.py", target="file:c.py", edge_type="IMPORTS"),
        ),
    )


def test_graph_ranking_uses_real_edges_and_respects_depth() -> None:
    documents = tuple(_file(name) for name in ("a", "b", "c", "unconnected"))
    seeds = (_ranked(documents[0], 1),)

    depth_one = graph_candidate_ranking(seeds, documents, _graph(), depth=1)
    depth_two = graph_candidate_ranking(seeds, documents, _graph(), depth=2)

    assert [item.file_path for item in depth_one] == ["b.py"]
    assert [item.file_path for item in depth_two] == ["b.py", "c.py"]
    assert all(item.file_path != "unconnected.py" for item in depth_two)


def test_graph_ranking_projects_symbol_evidence_to_file() -> None:
    document = _file("b")
    seed = _file("a")
    symbol = GraphNode(
        node_id="function:b.py:b:1",
        node_type="function",
        name="b",
        file_path="b.py",
        qualified_name="b.b",
        start_line=1,
        end_line=2,
    )
    graph = RepositoryGraph(
        nodes=(
            GraphNode(node_id="file:a.py", node_type="file", name="a.py", file_path="a.py"),
            symbol,
        ),
        edges=(GraphEdge(source="file:a.py", target=symbol.node_id, edge_type="CALLS"),),
    )
    ranking = graph_candidate_ranking((_ranked(seed, 1),), (seed, document), graph)
    assert [item.document_id for item in ranking] == ["file:b.py"]


def test_graph_augmented_ranking_is_stable_and_fuses_evidence() -> None:
    documents = tuple(_file(name) for name in ("a", "b", "c"))
    retrieval = (_ranked(documents[0], 1), _ranked(documents[2], 2))
    ranking = graph_augmented_ranking(retrieval, documents, _graph(), top_k=3)
    assert [item.document_id for item in ranking] == ["file:a.py", "file:b.py", "file:c.py"]
