"""Optional Neo4j persistence for an already-built Python RepositoryGraph."""

from __future__ import annotations

import importlib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from graphfix.graph.models import GraphEdge, GraphNode, RepositoryGraph

_NODE_LABELS = {
    "repository": "Repository",
    "directory": "Directory",
    "file": "File",
    "class": "Class",
    "function": "Function",
    "method": "Method",
}
_RELATIONSHIP_TYPES = frozenset({"CONTAINS", "DEFINED_IN", "IMPORTS", "CALLS", "INHERITS"})


class Neo4jResult(Protocol):
    """Minimal result surface used by the exporter."""

    def single(self, *, strict: bool = False) -> Mapping[str, Any] | None: ...


class Neo4jSession(Protocol):
    """Minimal session surface used by the exporter."""

    def __enter__(self) -> Neo4jSession: ...

    def __exit__(self, *args: object) -> None: ...

    def run(self, query: str, **parameters: Any) -> Neo4jResult: ...


class Neo4jDriver(Protocol):
    """Minimal driver surface accepted by :class:`Neo4jGraphExporter`."""

    def session(self, *, database: str) -> Neo4jSession: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Neo4jExportSummary:
    """Counts observed after exporting and verifying one RepositoryGraph."""

    python_nodes: int
    python_edges: int
    neo4j_nodes: int
    neo4j_relationships: int

    @property
    def parity_ok(self) -> bool:
        """Return whether Neo4j contains every exported node and relationship exactly once."""
        return (
            self.python_nodes == self.neo4j_nodes and self.python_edges == self.neo4j_relationships
        )


def node_to_neo4j(node: GraphNode) -> dict[str, Any]:
    """Convert one GraphFix node without inventing or retaining absent properties."""
    properties: dict[str, Any] = {
        "node_id": node.node_id,
        "name": node.name,
        "node_type": node.node_type,
    }
    for key, value in (
        ("path", node.file_path),
        ("qualified_name", node.qualified_name),
        ("start_line", node.start_line),
        ("end_line", node.end_line),
    ):
        if value is not None:
            properties[key] = value
    return {"label": _NODE_LABELS[node.node_type], "properties": properties}


def relationship_to_neo4j(edge: GraphEdge) -> dict[str, str]:
    """Convert one directed GraphFix edge to stable endpoint and type values."""
    return {"source": edge.source, "target": edge.target, "edge_type": edge.edge_type}


def batched(values: Sequence[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """Yield deterministic non-empty batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), batch_size):
        yield list(values[start : start + batch_size])


def load_neo4j_driver(uri: str, user: str, password: str) -> Neo4jDriver:
    """Load the optional Neo4j driver only when an export is explicitly requested."""
    try:
        module = importlib.import_module("neo4j")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Neo4j export requires the optional dependency. "
            "Install it with: python -m pip install -e .[neo4j]"
        ) from error
    graph_database = cast(Any, module).GraphDatabase
    return cast(Neo4jDriver, graph_database.driver(uri, auth=(user, password)))


class Neo4jGraphExporter:
    """Batch-export an immutable RepositoryGraph with idempotent Cypher MERGE operations."""

    def __init__(
        self,
        driver: Neo4jDriver,
        *,
        database: str = "neo4j",
        batch_size: int = 1_000,
    ) -> None:
        if not database:
            raise ValueError("database must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._driver = driver
        self._database = database
        self._batch_size = batch_size

    @staticmethod
    def _validate(graph: RepositoryGraph) -> None:
        node_ids = [node.node_id for node in graph.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("RepositoryGraph contains duplicate node IDs")
        known_ids = set(node_ids)
        edge_keys = [(edge.source, edge.target, edge.edge_type) for edge in graph.edges]
        if len(edge_keys) != len(set(edge_keys)):
            raise ValueError("RepositoryGraph contains duplicate relationships")
        missing = {
            endpoint
            for edge in graph.edges
            for endpoint in (edge.source, edge.target)
            if endpoint not in known_ids
        }
        if missing:
            raise ValueError(f"RepositoryGraph edges reference missing nodes: {sorted(missing)!r}")

    def export(self, graph: RepositoryGraph) -> Neo4jExportSummary:
        """Export and verify one graph without modifying the Python object."""
        self._validate(graph)
        nodes_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for node in graph.nodes:
            converted = node_to_neo4j(node)
            nodes_by_label[cast(str, converted["label"])].append(
                cast(dict[str, Any], converted["properties"])
            )
        edges_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in graph.edges:
            converted_edge = relationship_to_neo4j(edge)
            edges_by_type[converted_edge["edge_type"]].append(converted_edge)

        with self._driver.session(database=self._database) as session:
            session.run(
                "CREATE CONSTRAINT graphfix_node_id IF NOT EXISTS "
                "FOR (n:GraphFixNode) REQUIRE n.node_id IS UNIQUE"
            )
            for label in sorted(nodes_by_label):
                query = (
                    "UNWIND $rows AS row "
                    "MERGE (n:GraphFixNode {node_id: row.node_id}) "
                    f"SET n:{label} SET n += row"
                )
                for batch in batched(nodes_by_label[label], self._batch_size):
                    session.run(query, rows=batch)
            for edge_type in sorted(edges_by_type):
                if edge_type not in _RELATIONSHIP_TYPES:
                    raise ValueError(f"Unsupported relationship type: {edge_type}")
                query = (
                    "UNWIND $rows AS row "
                    "MATCH (source:GraphFixNode {node_id: row.source}) "
                    "MATCH (target:GraphFixNode {node_id: row.target}) "
                    f"MERGE (source)-[:{edge_type}]->(target)"
                )
                for batch in batched(edges_by_type[edge_type], self._batch_size):
                    session.run(query, rows=batch)

            neo4j_nodes = self._count_nodes(session, [node.node_id for node in graph.nodes])
            neo4j_relationships = sum(
                self._count_relationships(session, edge_type, rows)
                for edge_type, rows in sorted(edges_by_type.items())
            )

        return Neo4jExportSummary(
            python_nodes=len(graph.nodes),
            python_edges=len(graph.edges),
            neo4j_nodes=neo4j_nodes,
            neo4j_relationships=neo4j_relationships,
        )

    def _count_nodes(self, session: Neo4jSession, node_ids: list[str]) -> int:
        total = 0
        rows = [{"node_id": node_id} for node_id in node_ids]
        for batch in batched(rows, self._batch_size):
            record = session.run(
                "UNWIND $rows AS row "
                "MATCH (n:GraphFixNode {node_id: row.node_id}) "
                "RETURN count(DISTINCT n) AS count",
                rows=batch,
            ).single(strict=True)
            assert record is not None
            total += int(record["count"])
        return total

    def _count_relationships(
        self,
        session: Neo4jSession,
        edge_type: str,
        rows: list[dict[str, Any]],
    ) -> int:
        if edge_type not in _RELATIONSHIP_TYPES:
            raise ValueError(f"Unsupported relationship type: {edge_type}")
        total = 0
        query = (
            "UNWIND $rows AS row "
            "MATCH (source:GraphFixNode {node_id: row.source})"
            f"-[relationship:{edge_type}]->"
            "(target:GraphFixNode {node_id: row.target}) "
            "RETURN count(relationship) AS count"
        )
        for batch in batched(rows, self._batch_size):
            record = session.run(query, rows=batch).single(strict=True)
            assert record is not None
            total += int(record["count"])
        return total
