"""Focused tests for optional, idempotent RepositoryGraph export to Neo4j."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from graphfix.graph.models import GraphEdge, GraphNode, RepositoryGraph
from graphfix.graph.neo4j_export import (
    Neo4jGraphExporter,
    batched,
    load_neo4j_driver,
    node_to_neo4j,
    relationship_to_neo4j,
)


class FakeResult:
    """Return a count derived from the submitted verification batch."""

    def __init__(self, count: int = 0) -> None:
        self.count = count

    def single(self, *, strict: bool = False) -> Mapping[str, Any] | None:
        del strict
        return {"count": self.count}


class FakeSession:
    """Record Cypher calls while emulating successful parity counts."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def run(self, query: str, **parameters: Any) -> FakeResult:
        self.calls.append((query, parameters))
        count = len(parameters.get("rows", [])) if "RETURN count" in query else 0
        return FakeResult(count)


class FakeDriver:
    """Expose one reusable fake session."""

    def __init__(self) -> None:
        self.fake_session = FakeSession()

    def session(self, *, database: str) -> FakeSession:
        assert database == "neo4j"
        return self.fake_session

    def close(self) -> None:
        return None


def sample_graph() -> RepositoryGraph:
    """Create a graph with a stable file/function relationship."""
    return RepositoryGraph(
        nodes=(
            GraphNode(
                node_id="file:benchmarks/benchmark_serving.py",
                node_type="file",
                name="benchmark_serving.py",
                file_path="benchmarks/benchmark_serving.py",
                qualified_name="benchmarks.benchmark_serving",
            ),
            GraphNode(
                node_id="function:benchmarks/benchmark_serving.py:benchmark:10",
                node_type="function",
                name="benchmark",
                file_path="benchmarks/benchmark_serving.py",
                qualified_name="benchmarks.benchmark_serving.benchmark",
                start_line=10,
                end_line=20,
            ),
        ),
        edges=(
            GraphEdge(
                source="function:benchmarks/benchmark_serving.py:benchmark:10",
                target="file:benchmarks/benchmark_serving.py",
                edge_type="DEFINED_IN",
            ),
        ),
    )


def test_node_conversion_preserves_identity_and_existing_metadata() -> None:
    converted = node_to_neo4j(sample_graph().nodes[1])

    assert converted["label"] == "Function"
    assert converted["properties"] == {
        "node_id": "function:benchmarks/benchmark_serving.py:benchmark:10",
        "name": "benchmark",
        "node_type": "function",
        "path": "benchmarks/benchmark_serving.py",
        "qualified_name": "benchmarks.benchmark_serving.benchmark",
        "start_line": 10,
        "end_line": 20,
    }


def test_relationship_conversion_preserves_direction_and_type() -> None:
    assert relationship_to_neo4j(sample_graph().edges[0]) == {
        "source": "function:benchmarks/benchmark_serving.py:benchmark:10",
        "target": "file:benchmarks/benchmark_serving.py",
        "edge_type": "DEFINED_IN",
    }


def test_export_queries_use_merge_and_are_repeatable() -> None:
    driver = FakeDriver()
    exporter = Neo4jGraphExporter(driver, batch_size=1)

    first = exporter.export(sample_graph())
    first_queries = list(driver.fake_session.calls)
    driver.fake_session.calls.clear()
    second = exporter.export(sample_graph())

    assert first == second
    assert first.parity_ok
    assert first_queries == driver.fake_session.calls
    mutation_queries = [query for query, _parameters in first_queries if "UNWIND" in query]
    assert all("MERGE" in query or "RETURN count" in query for query in mutation_queries)
    assert any(
        "MERGE (n:GraphFixNode {node_id: row.node_id})" in query for query in mutation_queries
    )
    assert any("MERGE (source)-[:DEFINED_IN]->(target)" in query for query in mutation_queries)


def test_duplicate_node_identity_is_rejected() -> None:
    node = sample_graph().nodes[0]
    graph = RepositoryGraph(nodes=(node, node), edges=())

    with pytest.raises(ValueError, match="duplicate node IDs"):
        Neo4jGraphExporter(FakeDriver()).export(graph)


def test_batching_is_stable_and_validates_size() -> None:
    values = [{"value": value} for value in range(3)]
    assert list(batched(values, 2)) == [values[:2], values[2:]]
    with pytest.raises(ValueError, match="positive"):
        list(batched(values, 0))


def test_missing_optional_driver_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> Any:
        raise ModuleNotFoundError("No module named 'neo4j'")

    monkeypatch.setattr("graphfix.graph.neo4j_export.importlib.import_module", missing)
    with pytest.raises(RuntimeError, match=r"pip install -e \.\[neo4j\]"):
        load_neo4j_driver("bolt://localhost:7687", "neo4j", "secret")
