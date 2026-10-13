"""
Purpose:
    Builds the heterogeneous static repository graph.

Input:
    A ParsedRepository and repository name.

Output:
    Sorted nodes and conservative static edges.

Pipeline stage:
    Graph construction after parsing.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from graphfix.graph.models import EdgeType, GraphEdge, GraphNode, RepositoryGraph
from graphfix.parsing import ParsedRepository, SourceSymbol


def _directory_id(path: str) -> str:
    """
    Purpose:
        Build the stable graph ID for a directory node.

    Args:
        path: The file or directory path to use.

    Returns:
        A `str` result produced by the operation.

    Pipeline role:
        Static repository graph construction and inspection.
    """
    return f"directory:{path or '.'}"


def _file_id(path: str) -> str:
    """
    Purpose:
        Build the stable graph ID for a file node.

    Args:
        path: The file or directory path to use.

    Returns:
        A `str` result produced by the operation.

    Pipeline role:
        Static repository graph construction and inspection.
    """
    return f"file:{path}"


def _resolve_symbol_reference(
    reference: str,
    *,
    caller: SourceSymbol,
    by_qualified: dict[str, SourceSymbol],
    by_name: dict[str, tuple[SourceSymbol, ...]],
) -> SourceSymbol | None:
    """
    Purpose:
        Match a short static reference to a parsed symbol when the match is unambiguous.

    Args:
        reference: The reference value used by this operation.
        caller: The caller value used by this operation.
        by_qualified: The by qualified value used by this operation.
        by_name: The by name value used by this operation.

    Returns:
        A `SourceSymbol | None` result produced by the operation.

    Pipeline role:
        Static repository graph construction and inspection.
    """
    candidates = [reference]
    if reference.startswith("self.") and caller.parent:
        candidates.insert(0, f"{caller.module}.{caller.parent}.{reference.removeprefix('self.')}")
    candidates.extend((f"{caller.module}.{reference}", reference))
    for candidate in candidates:
        if candidate in by_qualified:
            return by_qualified[candidate]
    short_name = reference.rsplit(".", 1)[-1]
    matches = by_name.get(short_name, ())
    same_file = [symbol for symbol in matches if symbol.file_path == caller.file_path]
    if len(same_file) == 1:
        return same_file[0]
    return matches[0] if len(matches) == 1 else None


def build_repository_graph(parsed: ParsedRepository, *, repository_name: str) -> RepositoryGraph:
    """Construct nodes and statically resolvable edges in stable sorted order."""
    nodes: dict[str, GraphNode] = {
        "repository": GraphNode(node_id="repository", node_type="repository", name=repository_name)
    }
    edges: set[tuple[str, str, EdgeType]] = set()

    for parsed_file in parsed.files:
        path = PurePosixPath(parsed_file.file_path)
        classes_by_name = {
            symbol.name: symbol.symbol_id
            for symbol in parsed_file.symbols
            if symbol.kind == "class"
        }
        parents = list(path.parents)[:-1]
        previous = "repository"
        for parent in reversed(parents):
            directory = parent.as_posix()
            node_id = _directory_id(directory)
            nodes[node_id] = GraphNode(node_id=node_id, node_type="directory", name=parent.name)
            edges.add((previous, node_id, "CONTAINS"))
            previous = node_id
        file_node = _file_id(parsed_file.file_path)
        nodes[file_node] = GraphNode(
            node_id=file_node,
            node_type="file",
            name=path.name,
            file_path=parsed_file.file_path,
            qualified_name=parsed_file.module,
        )
        edges.add((previous, file_node, "CONTAINS"))
        for symbol in parsed_file.symbols:
            nodes[symbol.symbol_id] = GraphNode(
                node_id=symbol.symbol_id,
                node_type=symbol.kind,
                name=symbol.name,
                file_path=symbol.file_path,
                qualified_name=symbol.qualified_name,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
            )
            edges.add((symbol.symbol_id, file_node, "DEFINED_IN"))
            container = file_node
            if symbol.kind == "method" and symbol.parent:
                container = classes_by_name.get(symbol.parent, file_node)
            edges.add((container, symbol.symbol_id, "CONTAINS"))

    module_files = {
        parsed_file.module: _file_id(parsed_file.file_path) for parsed_file in parsed.files
    }
    for parsed_file in parsed.files:
        source = _file_id(parsed_file.file_path)
        for imported in parsed_file.imports:
            module = imported.module.lstrip(".")
            targets = sorted(
                node_id
                for name, node_id in module_files.items()
                if name == module or name.startswith(f"{module}.")
            )
            if targets:
                edges.add((source, targets[0], "IMPORTS"))

    symbols = parsed.symbols
    by_qualified = {symbol.qualified_name: symbol for symbol in symbols}
    grouped_by_name: dict[str, list[SourceSymbol]] = {}
    for symbol in symbols:
        grouped_by_name.setdefault(symbol.name, []).append(symbol)
    by_name = {
        name: tuple(grouped_by_name[name]) for name in sorted(grouped_by_name)
    }
    for symbol in symbols:
        for call in symbol.calls:
            target = _resolve_symbol_reference(
                call, caller=symbol, by_qualified=by_qualified, by_name=by_name
            )
            if target and target.symbol_id != symbol.symbol_id:
                edges.add((symbol.symbol_id, target.symbol_id, "CALLS"))
        if symbol.kind == "class":
            for base in symbol.bases:
                target = _resolve_symbol_reference(
                    base, caller=symbol, by_qualified=by_qualified, by_name=by_name
                )
                if target and target.kind == "class":
                    edges.add((symbol.symbol_id, target.symbol_id, "INHERITS"))

    graph_nodes = tuple(nodes[key] for key in sorted(nodes))
    graph_edges = tuple(
        GraphEdge(source=source, target=target, edge_type=edge_type)
        for source, target, edge_type in sorted(
            edges, key=lambda value: (value[2], value[0], value[1])
        )
    )
    return RepositoryGraph(nodes=graph_nodes, edges=graph_edges)
