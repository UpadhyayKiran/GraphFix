"""
Purpose:
    Defines graph nodes, edges, statistics, and queries.

Input:
    Deterministic graph nodes and edges.

Output:
    A serializable, inspectable RepositoryGraph.

Pipeline stage:
    Structural representation; not BM25 input.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict

NodeType = Literal["repository", "directory", "file", "class", "function", "method"]
EdgeType = Literal["CONTAINS", "DEFINED_IN", "IMPORTS", "CALLS", "INHERITS"]


class GraphModel(BaseModel):
    """
    Purpose:
        Base model that rejects unexpected graph fields.

    Pipeline role:
        Static repository graph construction and inspection.
    """
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class GraphNode(GraphModel):
    """
    Purpose:
        Represents one typed repository graph node.

    Contains:
        node_id: The node id value.
        node_type: The node type value.
        name: The name value.
        file_path: The file path value.
        qualified_name: The qualified name value.
        start_line: The start line value.
        end_line: The end line value.

    Pipeline role:
        Static repository graph construction and inspection.
    """
    node_id: str
    node_type: NodeType
    name: str
    file_path: str | None = None
    qualified_name: str | None = None
    start_line: int | None = None
    end_line: int | None = None


class GraphEdge(GraphModel):
    """
    Purpose:
        Represents one typed directed graph relationship.

    Contains:
        source: The source value.
        target: The target value.
        edge_type: The edge type value.

    Pipeline role:
        Static repository graph construction and inspection.
    """
    source: str
    target: str
    edge_type: EdgeType


class GraphStats(GraphModel):
    """
    Purpose:
        Stores node and edge totals grouped by type.

    Contains:
        total_nodes: The total nodes value.
        nodes_by_type: The nodes by type value.
        total_edges: The total edges value.
        edges_by_type: The edges by type value.

    Pipeline role:
        Static repository graph construction and inspection.
    """
    total_nodes: int
    nodes_by_type: dict[str, int]
    total_edges: int
    edges_by_type: dict[str, int]


class RepositoryGraph(GraphModel):
    """Serializable graph with deterministic query helpers."""

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    @property
    def stats(self) -> GraphStats:
        """
        Purpose:
            Count graph nodes and edges by type.

        Returns:
            A `GraphStats` result produced by the operation.

        Pipeline role:
            Static repository graph construction and inspection.
        """
        node_counts = Counter(node.node_type for node in self.nodes)
        edge_counts = Counter(edge.edge_type for edge in self.edges)
        return GraphStats(
            total_nodes=len(self.nodes),
            nodes_by_type=dict(sorted(node_counts.items())),
            total_edges=len(self.edges),
            edges_by_type=dict(sorted(edge_counts.items())),
        )

    def get_node(self, node_id: str) -> GraphNode | None:
        """
        Purpose:
            Return a graph node by ID, if it exists.

        Args:
            node_id: The graph node ID.

        Returns:
            A `GraphNode | None` result produced by the operation.

        Pipeline role:
            Static repository graph construction and inspection.
        """
        return next((node for node in self.nodes if node.node_id == node_id), None)

    def neighbors(self, node_id: str, edge_type: EdgeType | None = None) -> tuple[GraphNode, ...]:
        """
        Purpose:
            Return nodes connected to a node, optionally restricted to one edge type.

        Args:
            node_id: The graph node ID.
            edge_type: Optional edge type used to filter relationships.

        Returns:
            A `tuple[GraphNode, ...]` result produced by the operation.

        Pipeline role:
            Static repository graph construction and inspection.
        """
        ids: set[str] = set()
        for edge in self.edges:
            if edge_type is not None and edge.edge_type != edge_type:
                continue
            if edge.source == node_id:
                ids.add(edge.target)
            elif edge.target == node_id:
                ids.add(edge.source)
        return tuple(node for node in self.nodes if node.node_id in ids)

    def outgoing(self, node_id: str, edge_type: EdgeType) -> tuple[GraphNode, ...]:
        """
        Purpose:
            Return outgoing edges, optionally restricted to one edge type.

        Args:
            node_id: The graph node ID.
            edge_type: Optional edge type used to filter relationships.

        Returns:
            A `tuple[GraphNode, ...]` result produced by the operation.

        Pipeline role:
            Static repository graph construction and inspection.
        """
        ids = {
            edge.target
            for edge in self.edges
            if edge.source == node_id and edge.edge_type == edge_type
        }
        return tuple(node for node in self.nodes if node.node_id in ids)

    def callers(self, node_id: str) -> tuple[GraphNode, ...]:
        """
        Purpose:
            Return symbols with CALLS edges into the requested symbol.

        Args:
            node_id: The graph node ID.

        Returns:
            A `tuple[GraphNode, ...]` result produced by the operation.

        Pipeline role:
            Static repository graph construction and inspection.
        """
        ids = {
            edge.source
            for edge in self.edges
            if edge.target == node_id and edge.edge_type == "CALLS"
        }
        return tuple(node for node in self.nodes if node.node_id in ids)

    def callees(self, node_id: str) -> tuple[GraphNode, ...]:
        """
        Purpose:
            Return symbols called by the requested symbol.

        Args:
            node_id: The graph node ID.

        Returns:
            A `tuple[GraphNode, ...]` result produced by the operation.

        Pipeline role:
            Static repository graph construction and inspection.
        """
        return self.outgoing(node_id, "CALLS")

    def imports(self, node_id: str) -> tuple[GraphNode, ...]:
        """
        Purpose:
            Return modules or files reached by IMPORTS edges.

        Args:
            node_id: The graph node ID.

        Returns:
            A `tuple[GraphNode, ...]` result produced by the operation.

        Pipeline role:
            Static repository graph construction and inspection.
        """
        return self.outgoing(node_id, "IMPORTS")

    def bases(self, node_id: str) -> tuple[GraphNode, ...]:
        """
        Purpose:
            Return base-class nodes reached by INHERITS edges.

        Args:
            node_id: The graph node ID.

        Returns:
            A `tuple[GraphNode, ...]` result produced by the operation.

        Pipeline role:
            Static repository graph construction and inspection.
        """
        return self.outgoing(node_id, "INHERITS")
