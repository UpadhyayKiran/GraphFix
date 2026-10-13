"""
Purpose:
    Show the implemented Review-1 pipeline as a small local Streamlit walkthrough.

Input:
    One selected instance from the local 560-instance Loc-Bench snapshot.

Output:
    Read-only views of reconstruction, parsing, graph, BM25 documents, rankings, and metrics.

Pipeline stage:
    Review and demonstration layer around the existing GraphFix APIs.

Run example:
    py -3.11 -m streamlit run scripts/review1_explorer.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import streamlit as st
from datasets import load_from_disk  # type: ignore[import-untyped]

from graphfix.datasets import LocBenchDataset, normalize_locbench_records
from graphfix.datasets.models import GraphFixInput, GraphFixInstance
from graphfix.evaluation import evaluate_rankings, extract_gold_locations
from graphfix.graph import GraphEdge, RepositoryGraph, build_repository_graph
from graphfix.parsing import ParsedFile, ParsedRepository, PythonRepositoryParser
from graphfix.pipeline import BM25LocalizationPipeline, LocalizationResult, RetrievalConfig
from graphfix.repositories import RepositoryCheckout, RepositoryManager, RepositoryRequest
from graphfix.retrieval import (
    SearchDocument,
    build_file_documents,
    build_function_documents,
    tokenize,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = PROJECT_ROOT / "data" / "raw" / "locbench_v1" / "dataset"
REPOSITORY_CACHE = PROJECT_ROOT / "data" / "cache" / "repositories"
DEFAULT_INSTANCE = "UXARRAY__uxarray-1117"
EDGE_TYPES = ("CONTAINS", "DEFINED_IN", "IMPORTS", "CALLS", "INHERITS")
MAX_GRAPH_EDGES = 60


@st.cache_data(show_spinner=False)
def load_local_dataset() -> LocBenchDataset:
    """Load and normalize the local snapshot once for instance selection."""
    raw = load_from_disk(str(SNAPSHOT))
    return normalize_locbench_records(
        cast(Any, raw),
        raw_fields=cast(list[str], raw.column_names),
    )


def find_instance(dataset: LocBenchDataset, instance_id: str) -> GraphFixInstance:
    """Return the selected normalized instance or raise a readable lookup error."""
    for instance in dataset.instances:
        if instance.input.instance_id == instance_id:
            return instance
    raise ValueError(f"Unknown Loc-Bench instance: {instance_id}")


@st.cache_resource(show_spinner=False)
def reconstruct_and_prepare(
    instance_id: str,
    repository: str,
    base_commit: str,
) -> tuple[RepositoryCheckout, ParsedRepository, RepositoryGraph]:
    """Use existing GraphFix APIs to reconstruct, parse, and build the graph."""
    request = RepositoryRequest(
        instance_id=instance_id,
        repo=repository,
        base_commit=base_commit,
    )
    checkout = RepositoryManager(REPOSITORY_CACHE).reconstruct(request)
    if not checkout.success or not checkout.checkout_path or not checkout.resolved_commit:
        message = checkout.failure.message if checkout.failure else "unknown reconstruction failure"
        raise RuntimeError(message)
    if checkout.resolved_commit.casefold() != base_commit.casefold():
        raise RuntimeError("Checked-out HEAD does not match the requested base_commit")
    parsed = PythonRepositoryParser().parse(Path(checkout.checkout_path))
    graph = build_repository_graph(parsed, repository_name=repository)
    return checkout, parsed, graph


@st.cache_resource(show_spinner=False)
def run_localization(
    instance_id: str,
    repository: str,
    base_commit: str,
    problem_statement: str,
    category: str,
) -> LocalizationResult:
    """Run the existing gold-free BM25 localization pipeline and cache its result."""
    issue = GraphFixInput(
        instance_id=instance_id,
        repo=repository,
        base_commit=base_commit,
        problem_statement=problem_statement,
        category=category,
    )
    pipeline = BM25LocalizationPipeline(
        RepositoryManager(REPOSITORY_CACHE),
        config=RetrievalConfig(top_k=10),
    )
    return pipeline.localize(issue)


def symbol_rows(parsed_file: ParsedFile, kind: str | None = None) -> list[dict[str, Any]]:
    """Convert selected extracted symbols into a small display table."""
    return [
        {
            "name": symbol.name,
            "qualified_name": symbol.qualified_name,
            "signature": symbol.signature,
            "line_range": f"{symbol.start_line}-{symbol.end_line}",
        }
        for symbol in parsed_file.symbols
        if kind is None or symbol.kind == kind
    ]


def document_label(document: SearchDocument) -> str:
    """Return a concise select-box label for a real BM25 search document."""
    return document.qualified_name or document.file_path


def ranking_rows(localization: LocalizationResult, *, functions: bool) -> list[dict[str, Any]]:
    """Convert existing ranked candidates into file or function display rows."""
    rankings = localization.function_rankings if functions else localization.file_rankings
    if functions:
        return [
            {
                "rank": item.rank,
                "function": item.qualified_name or item.symbol_name,
                "file": item.file_path,
                "line_span": (
                    f"{item.start_line}-{item.end_line}" if item.start_line is not None else ""
                ),
                "BM25 score": item.score,
            }
            for item in rankings
        ]
    return [
        {"rank": item.rank, "file": item.file_path, "BM25 score": item.score}
        for item in rankings
    ]


def graph_seed_options(graph: RepositoryGraph, file_path: str) -> list[str]:
    """Prefer the selected file and its symbols while still allowing any graph node."""
    preferred = [
        node.node_id
        for node in graph.nodes
        if node.file_path == file_path or node.node_id == f"file:{file_path}"
    ]
    remaining = [node.node_id for node in graph.nodes if node.node_id not in set(preferred)]
    return preferred + remaining


def predicted_graph_seed(
    graph: RepositoryGraph,
    localization: LocalizationResult,
    predicted_file: str,
) -> str:
    """Choose a graph seed from the gold-free function ranking, then the file ranking."""
    if localization.function_rankings:
        function_node = graph.get_node(localization.function_rankings[0].document_id)
        if function_node is not None:
            return function_node.node_id
    file_node_id = f"file:{predicted_file}"
    if graph.get_node(file_node_id) is not None:
        return file_node_id
    options = graph_seed_options(graph, predicted_file)
    return options[0]


def neighborhood_edges(
    graph: RepositoryGraph,
    seed: str,
    depth: int,
    selected_types: set[str],
) -> tuple[list[GraphEdge], bool]:
    """Collect actual graph edges within one or two hops and cap the display size."""
    frontier = {seed}
    visited = {seed}
    collected: list[GraphEdge] = []
    for _ in range(depth):
        next_frontier: set[str] = set()
        for edge in graph.edges:
            if edge.edge_type not in selected_types:
                continue
            if edge.source in frontier or edge.target in frontier:
                collected.append(edge)
                next_frontier.update((edge.source, edge.target))
        frontier = next_frontier - visited
        visited.update(next_frontier)
        if not frontier:
            break
    unique = list(dict.fromkeys((edge.source, edge.target, edge.edge_type) for edge in collected))
    limited = unique[:MAX_GRAPH_EDGES]
    return (
        [
            GraphEdge(source=source, target=target, edge_type=edge_type)
            for source, target, edge_type in limited
        ],
        len(unique) > len(limited),
    )


def graphviz_source(graph: RepositoryGraph, edges: list[GraphEdge], seed: str) -> str:
    """Render selected real nodes and edges as a small Graphviz DOT graph."""
    node_ids = {seed}
    for edge in edges:
        node_ids.update((edge.source, edge.target))
    nodes = {node.node_id: node for node in graph.nodes if node.node_id in node_ids}
    lines = ["digraph GraphFixPipeline {", "rankdir=LR;", 'node [shape=box, style="rounded"];']
    for node_id in sorted(node_ids):
        node = nodes.get(node_id)
        label = f"{node.node_type}: {node.name}" if node else node_id
        attributes = f'label={json.dumps(label)}'
        if node_id == seed:
            attributes += ', style="rounded,filled", fillcolor="lightblue"'
        lines.append(f"{json.dumps(node_id)} [{attributes}];")
    for edge in edges:
        lines.append(
            f"{json.dumps(edge.source)} -> {json.dumps(edge.target)} "
            f"[label={json.dumps(edge.edge_type)}];"
        )
    lines.append("}")
    return "\n".join(lines)


def show_level_metrics(title: str, metrics: Any) -> None:
    """Display one existing LevelMetrics object without recalculating any value."""
    st.markdown(f"**{title}**")
    columns = st.columns(6)
    columns[0].metric("Hit@1", f"{metrics.hit_at_1:.4f}")
    columns[1].metric("Recall@5", f"{metrics.recall_at_5:.4f}")
    columns[2].metric("Recall@10", f"{metrics.recall_at_10:.4f}")
    columns[3].metric("MRR", f"{metrics.mrr:.4f}")
    columns[4].metric("Gold count", metrics.gold_count)
    columns[5].metric("First relevant rank", metrics.first_relevant_rank or "—")


def main() -> None:
    """Build the GraphFix walkthrough and keep gold below the localization boundary."""
    st.set_page_config(page_title="GraphFix Localization Explorer", layout="wide")
    st.title("GraphFix Localization Explorer")
    st.caption("Visualization of the implemented GraphFix localization pipeline.")
    st.info(
        "The RepositoryGraph is built and inspectable. "
        "BM25 does not use graph edges for ranking."
    )

    try:
        dataset = load_local_dataset()
    except Exception as error:
        st.error(f"Unable to load the local Loc-Bench snapshot: {error}")
        st.stop()

    instance_ids = [item.input.instance_id for item in dataset.instances]
    default_index = instance_ids.index(DEFAULT_INSTANCE) if DEFAULT_INSTANCE in instance_ids else 0
    selected_id = st.selectbox("Select Loc-Bench instance ID", instance_ids, index=default_index)
    instance = find_instance(dataset, selected_id)
    issue = instance.input

    st.header("1 — Loc-Bench input")
    st.write("Only inference-safe `GraphFixInput` fields are shown here.")
    st.json(issue.model_dump())
    st.markdown("**Problem statement**")
    st.write(issue.problem_statement)

    try:
        with st.spinner("Reconstructing the exact commit and preparing repository views..."):
            checkout, parsed, graph = reconstruct_and_prepare(
                issue.instance_id, issue.repo, issue.base_commit
            )
            localization = run_localization(
                issue.instance_id,
                issue.repo,
                issue.base_commit,
                issue.problem_statement,
                issue.category,
            )
    except Exception as error:
        st.error(f"Unable to prepare this instance: {error}")
        st.stop()

    st.header("2 — Exact repository reconstruction")
    st.code(
        f"Expected base_commit\n{issue.base_commit}\n"
        f"        ↓\nRepository reconstruction\n        ↓\n"
        f"Checked-out HEAD\n{checkout.resolved_commit}"
    )
    commit_matches = (
        checkout.resolved_commit
        and checkout.resolved_commit.casefold() == issue.base_commit.casefold()
    )
    if commit_matches:
        st.success("✓ Exact commit verified")
    else:
        st.error("Checked-out HEAD does not match base_commit")

    st.header("3 — Python AST parsing")
    stats = parsed.stats
    metric_columns = st.columns(6)
    for column, label, value in zip(
        metric_columns,
        ("Files scanned", "Python parsed", "Parser failures", "Classes", "Functions", "Methods"),
        (
            stats.files_scanned,
            stats.python_files_parsed,
            stats.parse_failures,
            stats.classes,
            stats.functions,
            stats.methods,
        ),
        strict=True,
    ):
        column.metric(label, value)

    parsed_paths = [item.file_path for item in parsed.files]
    predicted_file = localization.file_rankings[0].file_path
    parsed_file_index = parsed_paths.index(predicted_file) if predicted_file in parsed_paths else 0
    selected_file_path = st.selectbox(
        "Select parsed file",
        parsed_paths,
        index=parsed_file_index,
    )
    selected_file = next(item for item in parsed.files if item.file_path == selected_file_path)
    rows = symbol_rows(selected_file)
    st.dataframe(rows, width="stretch", hide_index=True)
    if not rows:
        st.caption("This parsed file contains no class, function, or method definitions.")

    st.header("4 — ParsedRepository")
    classes = symbol_rows(selected_file, "class")
    functions = symbol_rows(selected_file, "function")
    methods = symbol_rows(selected_file, "method")
    st.json(
        {
            "file_path": selected_file.file_path,
            "module": selected_file.module,
            "import_count": len(selected_file.imports),
            "class_count": len(classes),
            "function_count": len(functions),
            "method_count": len(methods),
        }
    )
    for label, symbol_group in (
        ("Classes", classes),
        ("Functions", functions),
        ("Methods", methods),
    ):
        st.markdown(f"**{label}**")
        if symbol_group:
            st.dataframe(symbol_group, width="stretch", hide_index=True)
        else:
            st.caption(f"No {label.lower()} were extracted from this file.")
    with st.expander("Show complete ParsedFile details", expanded=False):
        st.json(selected_file.model_dump())

    st.header("5 — Repository graph")
    graph_stats = graph.stats
    graph_columns = st.columns(2)
    graph_columns[0].metric("Total nodes", graph_stats.total_nodes)
    graph_columns[1].metric("Total edges", graph_stats.total_edges)
    st.markdown("**Nodes by type**")
    st.dataframe(
        [{"node type": key, "count": value} for key, value in graph_stats.nodes_by_type.items()],
        width="stretch",
        hide_index=True,
    )
    st.markdown("**Edges by type**")
    st.dataframe(
        [{"edge type": key, "count": value} for key, value in graph_stats.edges_by_type.items()],
        width="stretch",
        hide_index=True,
    )
    seed_options = graph_seed_options(graph, selected_file.file_path)
    default_seed = predicted_graph_seed(graph, localization, predicted_file)
    seed_index = seed_options.index(default_seed)
    seed = st.selectbox("Graph seed node", seed_options, index=seed_index)
    depth = st.radio("Neighborhood depth", (1, 2), horizontal=True)
    selected_edge_types = set(
        st.multiselect("Edge types", EDGE_TYPES, default=list(EDGE_TYPES))
    )
    edges, graph_was_limited = neighborhood_edges(graph, seed, depth, selected_edge_types)
    if edges:
        st.graphviz_chart(graphviz_source(graph, edges, seed), width="stretch")
        if graph_was_limited:
            st.caption(f"View limited to the first {MAX_GRAPH_EDGES} actual graph edges.")
    else:
        st.info("No matching graph edges were found around this seed.")

    st.header("6 — BM25 documents")
    file_documents = build_file_documents(parsed)
    function_documents = build_function_documents(parsed)
    document_kind = st.radio("Document type", ("File", "Function/method"), horizontal=True)
    documents = file_documents if document_kind == "File" else function_documents
    predicted_document_id = (
        localization.file_rankings[0].document_id
        if document_kind == "File"
        else localization.function_rankings[0].document_id
    )
    document_index = next(
        (
            index
            for index, item in enumerate(documents)
            if item.document_id == predicted_document_id
        ),
        0,
    )
    document = st.selectbox(
        "Search document",
        documents,
        index=document_index,
        format_func=document_label,
    )
    st.json(
        {
            "document_id": document.document_id,
            "kind": document.kind,
            "file_path": document.file_path,
            "qualified_name": document.qualified_name,
            "line_span": [document.start_line, document.end_line],
            "text_preview": document.text[:2000],
            "token_preview": list(tokenize(document.text)[:100]),
        }
    )

    st.header("7 — BM25 Localization Results")
    st.markdown("**BM25 Retrieval & Ranking**")
    st.caption("BM25 scores are lexical relevance scores, not probabilities or confidence values.")
    st.subheader("Ranked Candidate Files")
    st.dataframe(
        ranking_rows(localization, functions=False),
        width="stretch",
        hide_index=True,
    )
    st.subheader("Ranked Candidate Functions")
    st.dataframe(
        ranking_rows(localization, functions=True),
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.success("LOCALIZATION COMPLETE")
    st.markdown("### Gold data has not been used above this point.")
    st.divider()

    # This is the first point where the app reads GraphFixGold.
    gold = extract_gold_locations(instance.gold)
    metrics = evaluate_rankings(
        localization.file_rankings,
        localization.function_rankings,
        gold,
    )

    st.header("8 — Gold / evaluation only")
    st.write("**Gold files**", list(gold.files))
    st.write("**Gold functions**", list(gold.functions))
    show_level_metrics("File metrics", metrics.file)
    show_level_metrics("Function metrics", metrics.function)

    st.header("Review-1 status")
    st.markdown(
        """
**Implemented localization components:** exact repository reconstruction, AST parsing,
        RepositoryGraph construction, BM25 file/function baseline, and gold-safe evaluation.

        **Not implemented:** graph-informed ranking, dense/hybrid retrieval, graph traversal
agents, fine-tuned controllers, or calibrated early stopping.
        """
    )


if __name__ == "__main__":
    main()
