"""Offline pipeline explorer for the GraphFix Phase-1 final demonstration."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, cast

import streamlit as st
from datasets import load_from_disk  # type: ignore[import-untyped]

from graphfix.datasets import GraphFixGold
from graphfix.evaluation import evaluate_rankings, extract_gold_locations, normalize_repository_path
from graphfix.graph import GraphEdge, RepositoryGraph, build_repository_graph
from graphfix.parsing import ParsedRepository, PythonRepositoryParser, SourceSymbol
from graphfix.retrieval import (
    RankedCandidate,
    SearchDocument,
    build_file_documents,
    build_function_documents,
    tokenize,
)

ROOT = Path(__file__).resolve().parents[1]
PHASE1_SUMMARY_DIR = ROOT / "results" / "phase1_evaluation_summary"
FULL_COMPARISON = PHASE1_SUMMARY_DIR / "full560_method_comparison.csv"
COHORT = ROOT / "results" / "localization_benchmark" / "evaluation_cohort.json"
FULL_RESULTS = ROOT / "results" / "retrieval_comparison" / "full_evaluation"
FULL560_SUMMARY = (
    ROOT
    / "results"
    / "adaptive_navigation_experiments"
    / "fixed_pool_560_analysis"
    / "full560_summary.json"
)
ADAPTIVE_FIRST_100 = (
    ROOT / "results" / "adaptive_navigation_experiments" / "fixed_pool_100" / "adaptive"
)
ADAPTIVE_CONTINUATION = (
    ROOT / "results" / "adaptive_navigation_experiments" / "fixed_pool_101_560" / "adaptive"
)
DATASET_SNAPSHOT = ROOT / "data" / "raw" / "locbench_v1" / "dataset"
EXPANSION_DETAILS = (
    ROOT
    / "results"
    / "adaptive_navigation_experiments"
    / "expanded_pool"
    / "analysis"
    / "per_instance.jsonl"
)
DEFAULT_INSTANCE = "vllm-project__vllm-9390"
ALL_REPOSITORIES = "All repositories"
EDGE_TYPES = ("CONTAINS", "DEFINED_IN", "IMPORTS", "CALLS", "INHERITS")
NODE_TYPES = ("repository", "directory", "file", "class", "function", "method")
MAX_GRAPH_EDGES = 60


@st.cache_data
def load_json(path: Path) -> Any:
    """Load one precomputed UTF-8 JSON artifact."""
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


@st.cache_data
def load_csv(path: Path) -> list[dict[str, str]]:
    """Load one precomputed CSV artifact without a dataframe dependency."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@st.cache_data
def load_cohort_ids() -> tuple[str, ...]:
    """Load the authoritative ordered frozen evaluation cohort."""
    payload = cast(dict[str, Any], json.loads(COHORT.read_text(encoding="utf-8")))
    instance_ids = tuple(cast(list[str], payload["included_ids"]))
    if len(instance_ids) != 560 or len(set(instance_ids)) != 560:
        raise ValueError("Expected exactly 560 unique frozen cohort IDs")
    return instance_ids


def repository_from_instance_id(instance_id: str) -> str:
    """Recover the Loc-Bench owner/repository label from its canonical instance ID."""
    match = re.fullmatch(r"(.+)__(.+)-(\d+)", instance_id)
    if match is None:
        raise ValueError(f"Unexpected Loc-Bench instance ID: {instance_id}")
    return f"{match.group(1)}/{match.group(2)}"


def adaptive_artifact_path(instance_id: str) -> Path:
    """Resolve one definitive Adaptive Fixed artifact across its frozen partitions."""
    first = ADAPTIVE_FIRST_100 / f"{instance_id}.json"
    continuation = ADAPTIVE_CONTINUATION / f"{instance_id}.json"
    if first.is_file() and continuation.is_file():
        raise ValueError(f"Duplicate definitive Adaptive artifact: {instance_id}")
    if first.is_file():
        return first
    if continuation.is_file():
        return continuation
    raise FileNotFoundError(f"Definitive Adaptive artifact is absent: {instance_id}")


def method_artifact_path(instance_id: str, method: str) -> Path:
    """Resolve one frozen deterministic localization artifact."""
    return FULL_RESULTS / method / f"{instance_id}.json"


def local_checkout_path(saved_path: str) -> Path:
    """Prefer the package-local checkout when a frozen artifact stores another root."""
    saved = Path(saved_path)
    try:
        repositories_index = saved.parts.index("repositories")
    except ValueError:
        return saved
    packaged = ROOT / "data" / "cache" / Path(*saved.parts[repositories_index:])
    return packaged if packaged.is_dir() else saved


@st.cache_data(show_spinner=False)
def load_adaptive_result(instance_id: str) -> dict[str, Any]:
    """Load one definitive full-cohort Adaptive Fixed trace."""
    return cast(
        dict[str, Any],
        json.loads(adaptive_artifact_path(instance_id).read_text(encoding="utf-8")),
    )


@st.cache_data(show_spinner=False)
def load_method_result(instance_id: str, method: str) -> dict[str, Any]:
    """Load one frozen BM25, Dense, Hybrid, or Static Graph result."""
    return cast(
        dict[str, Any],
        json.loads(method_artifact_path(instance_id, method).read_text(encoding="utf-8")),
    )


@st.cache_data(show_spinner=False)
def available_demo_ids() -> tuple[str, ...]:
    """Retain cohort order while requiring the complete local ten-stage artifact set."""
    available: list[str] = []
    for instance_id in load_cohort_ids():
        try:
            adaptive_artifact_path(instance_id)
        except FileNotFoundError:
            continue
        if not all(
            method_artifact_path(instance_id, method).is_file()
            for method in ("bm25", "dense", "hybrid", "graph")
        ):
            continue
        available.append(instance_id)
    return tuple(available)


@st.cache_data(show_spinner=False)
def load_dataset_input(instance_id: str) -> dict[str, Any]:
    """Load only inference-safe Loc-Bench columns before the gold boundary."""
    dataset = load_from_disk(str(DATASET_SNAPSHOT)).select_columns(
        ["instance_id", "repo", "base_commit", "problem_statement", "category"]
    )
    rows = dataset.filter(lambda row: row["instance_id"] == instance_id)
    if len(rows) != 1:
        raise KeyError(f"Loc-Bench input is absent or duplicated: {instance_id}")
    return cast(dict[str, Any], rows[0])


@st.cache_data(show_spinner=False)
def load_dataset_gold(instance_id: str) -> GraphFixGold:
    """Load GraphFixGold only after localization has completed."""
    dataset = load_from_disk(str(DATASET_SNAPSHOT)).select_columns(
        ["instance_id", "patch", "test_patch", "labels", "edit_functions", "added_functions"]
    )
    rows = dataset.filter(lambda row: row["instance_id"] == instance_id)
    if len(rows) != 1:
        raise KeyError(f"Loc-Bench gold is absent or duplicated: {instance_id}")
    row = cast(dict[str, Any], rows[0])
    return GraphFixGold(
        patch=cast(str, row["patch"]),
        test_patch=cast(str, row["test_patch"]),
        labels=cast(list[str] | None, row["labels"]),
        edit_functions=cast(list[str], row["edit_functions"]),
        added_functions=cast(list[str], row["added_functions"]),
    )


def require_artifact(path: Path) -> bool:
    """Explain how to recover a missing summary instead of crashing."""
    if path.exists():
        return True
    st.error(f"Required precomputed artifact is absent: `{path.relative_to(ROOT)}`")
    st.code(r".\.venv\Scripts\python.exe scripts\analyze_adaptive_pilot.py")
    return False


@st.cache_resource(show_spinner=False, max_entries=3)
def prepare_repository_views(
    checkout_path: str,
    repository: str,
    requested_commit: str,
    verified_commit: str,
) -> tuple[
    ParsedRepository,
    RepositoryGraph,
    tuple[SearchDocument, ...],
    tuple[SearchDocument, ...],
]:
    """Parse an existing verified checkout and build read-only implementation views."""
    if requested_commit.casefold() != verified_commit.casefold():
        raise ValueError("Saved checkout commit does not match the requested commit")
    root = local_checkout_path(checkout_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Saved checkout is unavailable: {root}")
    parsed = PythonRepositoryParser().parse(root)
    graph = build_repository_graph(parsed, repository_name=repository)
    return (
        parsed,
        graph,
        build_file_documents(parsed),
        build_function_documents(parsed),
    )


@st.cache_data(show_spinner=False)
def load_expansion_details() -> dict[str, dict[str, Any]]:
    """Load the saved candidate-expansion analysis without rerunning navigation."""
    if not EXPANSION_DETAILS.exists():
        return {}
    rows = [
        cast(dict[str, Any], json.loads(line))
        for line in EXPANSION_DETAILS.read_text(encoding="utf-8").splitlines()
    ]
    return {row["instance_id"]: row for row in rows}


@st.cache_data(show_spinner=False)
def load_expansion_admissions(instance_id: str) -> dict[str, list[str]] | None:
    """Read only gold-free candidate-admission provenance for the pre-boundary view."""
    path = (
        ROOT
        / "results"
        / "adaptive_navigation_experiments"
        / "expanded_pool"
        / "adaptive"
        / f"{instance_id}.json"
    )
    if not path.exists():
        return None
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    return {
        "expanded_file_candidate_ids": list(
            payload["file_navigation"].get("expanded_candidate_ids", [])
        ),
        "expanded_function_candidate_ids": list(
            payload["function_navigation"].get("expanded_candidate_ids", [])
        ),
    }


def neighborhood_edges(
    graph: RepositoryGraph,
    seed: str,
    depth: int,
    edge_types: set[str],
    node_types: set[str],
) -> tuple[list[GraphEdge], bool]:
    """Collect a capped real subgraph around a seed using selected types and depth."""
    nodes = {node.node_id: node for node in graph.nodes}
    frontier = {seed}
    visited = {seed}
    collected: list[GraphEdge] = []
    for _ in range(depth):
        reached: set[str] = set()
        for edge in graph.edges:
            if edge.edge_type not in edge_types:
                continue
            if edge.source not in frontier and edge.target not in frontier:
                continue
            other = edge.target if edge.source in frontier else edge.source
            other_node = nodes.get(other)
            if other_node is None or other_node.node_type not in node_types:
                continue
            collected.append(edge)
            reached.update((edge.source, edge.target))
        frontier = reached - visited
        visited.update(reached)
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


def repository_root_id(graph: RepositoryGraph) -> str:
    """Return the graph's actual validated repository-node ID."""
    roots = [node.node_id for node in graph.nodes if node.node_type == "repository"]
    if not roots:
        raise ValueError("RepositoryGraph contains no repository node")
    root = roots[0]
    if graph.get_node(root) is None:
        raise ValueError(f"Repository root is not resolvable: {root}")
    return root


def require_graph_root(graph: RepositoryGraph, root: str) -> str:
    """Reject a graph-rendering root that is not an actual RepositoryGraph node."""
    if graph.get_node(root) is None:
        raise ValueError(f"Graph root is not present in RepositoryGraph: {root}")
    return root


def visible_node_ids(root: str, edges: list[GraphEdge]) -> set[str]:
    """Return exactly the root plus endpoints of displayed real edges."""
    return {
        root,
        *(edge.source for edge in edges),
        *(edge.target for edge in edges),
    }


def adaptive_connected_edges(
    graph: RepositoryGraph,
    target: str,
    previous: set[str],
    new: set[str],
) -> tuple[list[GraphEdge], bool]:
    """Build a connected, bounded real-edge view for one saved adaptive step."""
    require_graph_root(graph, target)
    relevant = {
        node_id for node_id in {target, *previous, *new} if graph.get_node(node_id) is not None
    }
    connected = {target}
    collected: list[GraphEdge] = []
    # Two bounded passes can connect relevant prior/new evidence through the target
    # neighborhood without injecting disconnected historical nodes into DOT.
    for _ in range(2):
        additions: list[GraphEdge] = []
        for edge in graph.edges:
            if edge.source not in connected and edge.target not in connected:
                continue
            other = edge.target if edge.source in connected else edge.source
            if other not in relevant and target not in {edge.source, edge.target}:
                continue
            additions.append(edge)
        if not additions:
            break
        for edge in additions:
            connected.update((edge.source, edge.target))
        collected.extend(additions)
    unique = list(dict.fromkeys((edge.source, edge.target, edge.edge_type) for edge in collected))
    limited = unique[:MAX_GRAPH_EDGES]
    return (
        [
            GraphEdge(source=source, target=target_id, edge_type=edge_type)
            for source, target_id, edge_type in limited
        ],
        len(unique) > len(limited),
    )


def stop_fallback_target(
    graph: RepositoryGraph, trace: list[dict[str, Any]], step_index: int
) -> str:
    """Resolve STOP to the latest prior valid target, then the actual repository root."""
    for prior in reversed(trace[:step_index]):
        candidate = prior.get("node_id")
        if isinstance(candidate, str) and graph.get_node(candidate) is not None:
            return candidate
    return repository_root_id(graph)


def graphviz_source(
    graph: RepositoryGraph,
    edges: list[GraphEdge],
    seed: str,
    *,
    previous: set[str] | None = None,
    new: set[str] | None = None,
    seed_color: str = "lightblue",
) -> str:
    """Render a bounded real subgraph with optional adaptive-step highlighting."""
    previous = previous or set()
    new = new or set()
    require_graph_root(graph, seed)
    node_ids = visible_node_ids(seed, edges)
    nodes = {node.node_id: node for node in graph.nodes if node.node_id in node_ids}
    lines = [
        "digraph GraphFixRepository {",
        "rankdir=LR;",
        'node [shape=box, style="rounded"];',
    ]
    for node_id in sorted(node_ids):
        node = nodes.get(node_id)
        label = f"{node.node_type}: {node.name}" if node else node_id
        attributes = f"label={json.dumps(label)}"
        if node_id == seed:
            attributes += f', style="rounded,filled", fillcolor={json.dumps(seed_color)}'
        elif node_id in new:
            attributes += ', style="rounded,filled", fillcolor="#8fd3c7"'
        elif node_id in previous:
            attributes += ', style="rounded,filled", fillcolor="#d9e2ec"'
        lines.append(f"{json.dumps(node_id)} [{attributes}];")
    for edge in edges:
        lines.append(
            f"{json.dumps(edge.source)} -> {json.dumps(edge.target)} "
            f"[label={json.dumps(edge.edge_type)}];"
        )
    lines.append("}")
    return "\n".join(lines)


def show_graph_sanity(graph: RepositoryGraph, root: str, edges: list[GraphEdge]) -> None:
    """Show compact, presentation-friendly facts immediately above a graph."""
    node = graph.get_node(require_graph_root(graph, root))
    assert node is not None
    columns = st.columns(4)
    columns[0].metric("Selected root", root)
    columns[1].metric("Root node type", node.node_type)
    columns[2].metric("Visible nodes", len(visible_node_ids(root, edges)))
    columns[3].metric("Visible edges", len(edges))


def show_empty_graph_context(
    graph: RepositoryGraph,
    root: str,
    edge_types: set[str],
    depth: int,
) -> None:
    """Explain a valid empty filtered neighborhood instead of rendering one box."""
    node = graph.get_node(require_graph_root(graph, root))
    assert node is not None
    st.info("No matching graph relationships for this node and current filters.")
    st.json(
        {
            "selected_root": root,
            "root_node_type": node.node_type,
            "relationship_types": sorted(edge_types),
            "depth": depth,
        },
        expanded=False,
    )


def node_relationship_rows(
    graph: RepositoryGraph, node_id: str, *, incoming: bool
) -> list[dict[str, str]]:
    """Return actual incoming or outgoing relations for one graph node."""
    rows = []
    for edge in graph.edges:
        matches = edge.target == node_id if incoming else edge.source == node_id
        if matches:
            rows.append(
                {
                    "Relationship": edge.edge_type,
                    "Connected node": edge.source if incoming else edge.target,
                }
            )
    return rows


def show_node_inspector(
    parsed: ParsedRepository,
    graph: RepositoryGraph,
    node_ids: list[str],
    *,
    key: str,
) -> None:
    """Render secondary metadata and relationships below a visible graph."""
    st.markdown("#### Inspect selected node")
    inspected_id = st.selectbox("Graph node", node_ids, key=key)
    inspected = graph.get_node(inspected_id)
    if inspected is None:
        st.error("Selected graph node is no longer available.")
        return
    symbol = symbol_for_node(parsed, inspected_id)
    metadata: dict[str, Any] = inspected.model_dump()
    if symbol is not None:
        metadata.update(
            {
                "signature": symbol.signature,
                "docstring_preview": symbol.docstring[:1000],
                "calls": list(symbol.calls),
                "bases": list(symbol.bases),
            }
        )
    st.json(metadata, expanded=False)
    if symbol is not None and symbol.source:
        st.code(symbol.source[:3000], language="python")
    incoming, outgoing = st.tabs(["Incoming relationships", "Outgoing relationships"])
    incoming.dataframe(
        node_relationship_rows(graph, inspected_id, incoming=True),
        width="stretch",
        hide_index=True,
    )
    outgoing.dataframe(
        node_relationship_rows(graph, inspected_id, incoming=False),
        width="stretch",
        hide_index=True,
    )


def symbol_for_node(parsed: ParsedRepository, node_id: str) -> SourceSymbol | None:
    """Resolve a graph symbol node to its parsed AST record."""
    return next((symbol for symbol in parsed.symbols if symbol.symbol_id == node_id), None)


def document_label(document: SearchDocument) -> str:
    """Return a compact real SearchDocument label."""
    return document.qualified_name or document.file_path


def candidate_rank_map(ranking: list[dict[str, Any]]) -> dict[str, int]:
    """Map saved candidate IDs to ranks without comparing method-specific scores."""
    return {candidate["document_id"]: int(candidate["rank"]) for candidate in ranking}


def navigation_view(navigation: dict[str, Any]) -> dict[str, Any]:
    """Derive gold-free presentation fields from one frozen navigation trace."""
    evidence: set[tuple[str, str, str, str]] = set()
    action_targets: set[tuple[str, str]] = set()
    unique_nodes: set[str] = set()
    trace: list[dict[str, Any]] = []
    for index, step in enumerate(cast(list[dict[str, Any]], navigation["trace"]), start=1):
        pair = (cast(str, step["action"]), cast(str, step["node_id"]))
        repeated = pair in action_targets
        action_targets.add(pair)
        additions = 0
        relationships: list[str] = []
        for observed in cast(list[dict[str, Any]], step["observed_nodes"]):
            item = (
                cast(str, observed["node_id"]),
                cast(str, observed["relationship"]),
                cast(str, observed["direction"]),
                cast(str, observed["snippet"]),
            )
            if item not in evidence:
                evidence.add(item)
                additions += 1
            unique_nodes.add(cast(str, observed["node_id"]))
            relationships.append(
                f"{observed['direction']} {observed['relationship']} -> {observed['node_id']}"
            )
        trace.append(
            {
                "step": index,
                "action": step["action"],
                "node_id": step["node_id"],
                "observation": "; ".join(relationships) or step["message"],
                "new_evidence": additions > 0,
                "repeated_action_target": repeated,
                "cumulative_unique_nodes": len(unique_nodes),
                "validation_errors": list(step["validation_errors"]),
            }
        )
    return {"trace": trace}


def selected_localization_view(instance_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assemble one inference-safe selected-instance view from definitive artifacts."""
    adaptive = load_adaptive_result(instance_id)
    if adaptive["adaptive_config"]["candidate_pool"] != "fixed":
        raise ValueError(f"Expected fixed-pool Adaptive artifact: {instance_id}")
    method_results = {
        method: load_method_result(instance_id, method)
        for method in ("bm25", "dense", "hybrid", "graph")
    }
    selected = {
        "instance_id": instance_id,
        "repository": adaptive["repository"],
        "issue": adaptive["issue"],
        "requested_commit": adaptive["base_commit"],
        "verified_commit": adaptive["checked_out_commit"],
        "checkout_path": adaptive["checkout_path"],
        "parser_stats": adaptive["parser_stats"],
        "graph_stats": adaptive["graph_stats"],
        "file_navigation": navigation_view(cast(dict[str, Any], adaptive["file_navigation"])),
        "function_navigation": navigation_view(
            cast(dict[str, Any], adaptive["function_navigation"])
        ),
        "rankings": {
            **{
                method: {
                    "file": result["file_rankings"],
                    "function": result["function_rankings"],
                }
                for method, result in method_results.items()
            },
            "adaptive": {
                "file": adaptive["file_rankings"],
                "function": adaptive["function_rankings"],
            },
        },
    }
    return selected, adaptive


def ranked_candidates(rows: list[dict[str, Any]]) -> tuple[RankedCandidate, ...]:
    """Validate persisted ranking dictionaries before post-hoc evaluation."""
    return tuple(RankedCandidate.model_validate(row) for row in rows)


def relevant_evidence(
    adaptive: dict[str, Any], gold_files: set[str], gold_functions: set[str]
) -> list[dict[str, Any]]:
    """Identify gold-relevant observed nodes strictly after the gold boundary."""
    found: dict[str, dict[str, Any]] = {}
    for level in ("file_navigation", "function_navigation"):
        for step in adaptive[level]["trace"]:
            for node in step["observed_nodes"]:
                path = normalize_repository_path(cast(str, node["file_path"]))
                levels: list[str] = []
                if path in gold_files:
                    levels.append("file")
                function = f"{path}:{str(node['name']).casefold()}"
                if node["node_type"] in ("function", "method") and function in gold_functions:
                    levels.append("function")
                if levels:
                    found[cast(str, node["node_id"])] = {
                        "node_id": node["node_id"],
                        "gold_levels": levels,
                    }
    return list(found.values())


def rank_takeaway(level: str, before: int | None, after: int | None) -> str:
    """Describe one selected-case rank transition without hiding adverse outcomes."""
    noun = "file" if level == "file" else "callable"
    if before is None and after is None:
        return f"No relevant {noun} appears in either Top-10 ranking."
    if before is None:
        return f"Adaptive brings the first relevant {noun} into the Top 10 at rank {after}."
    if after is None:
        return f"Adaptive moves the first relevant {noun} from rank {before} out of the Top 10."
    if after < before:
        return f"Adaptive moves the first relevant {noun} from rank {before} to rank {after}."
    if after > before:
        return f"Adaptive moves the first relevant {noun} from rank {before} to rank {after}."
    return f"The first relevant {noun} remains at rank {before}."


def metric_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Convert stored comparison rows into display-ready numeric metric records."""
    labels = {
        "bm25": "BM25",
        "dense": "Dense",
        "hybrid": "Hybrid",
        "graph": "Static Graph",
    }
    return [
        {
            "Method": labels[row["method"]],
            "Level": row["level"].title(),
            "Hit@1": float(row["hit_at_1"]),
            "Recall@5": float(row["recall_at_5"]),
            "Recall@10": float(row["recall_at_10"]),
            "MRR": float(row["mrr"]),
        }
        for row in rows
    ]


def ranking_table(ranking: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the presentation columns for a saved candidate ranking."""
    rows = []
    for candidate in ranking:
        row = {
            "Rank": candidate["rank"],
            "File": candidate["file_path"],
            "Score": round(float(candidate["score"]), 6),
        }
        if candidate.get("kind") != "file":
            row.update(
                {
                    "Callable": candidate.get("qualified_name")
                    or candidate.get("symbol_name")
                    or "—",
                    "Lines": (
                        f"{candidate['start_line']}–{candidate['end_line']}"
                        if candidate.get("start_line") is not None
                        and candidate.get("end_line") is not None
                        else "—"
                    ),
                }
            )
        rows.append(row)
    return rows


def trace_table(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return an auditable but compact view of saved navigation steps."""
    return [
        {
            "Step": step["step"],
            "Action": step["action"],
            "Target node": step["node_id"],
            "Observation / relation": step["observation"],
            "New evidence": "YES" if step["new_evidence"] else "NO",
            "Repeated action": "YES" if step["repeated_action_target"] else "NO",
            "Cumulative unique nodes": step["cumulative_unique_nodes"],
            "Validation": "; ".join(step["validation_errors"]) or "—",
        }
        for step in trace
    ]


def readable_node_id(node_id: str | None) -> str:
    """Return a concise trace label while keeping the persisted ID unchanged."""
    if not node_id:
        return "STOP"
    parts = node_id.split(":", 3)
    node_type = parts[0] if parts else "node"
    qualified = parts[2] if len(parts) > 2 else node_id
    suffix = "()" if node_type in {"function", "method"} else ""
    return f"{qualified.rsplit('.', 1)[-1]}{suffix}"


def stop_reason_text(reason: str) -> str:
    """Explain a persisted navigation stop code without shortening it in a metric."""
    return {
        "explicit_stop": "Controller selected STOP",
        "invalid_response_stop": "Retry limit reached after invalid controller responses",
        "max_steps": "Maximum navigation-step budget reached",
    }.get(reason, reason.replace("_", " ").title())


def rank_text(value: int | None) -> str:
    """Render a Top-10 rank without treating an unavailable complete rank as eleven."""
    return str(value) if value is not None else ">10 / not retrieved"


st.set_page_config(page_title="GraphFix Pipeline Walkthrough", layout="wide")
st.title("GraphFix Pipeline Walkthrough")
st.caption(
    "End-to-end software issue localization using retrieval, repository structure "
    "and adaptive graph navigation."
)

required = (COHORT, DATASET_SNAPSHOT, FULL_COMPARISON)
if not all(require_artifact(path) for path in required):
    st.stop()

full_rows = load_csv(FULL_COMPARISON)
instance_ids = available_demo_ids()
if not instance_ids:
    st.error("No complete local demo instances are available.")
    st.stop()
repositories = tuple(sorted({repository_from_instance_id(value) for value in instance_ids}))
selected_repository = st.sidebar.selectbox("Repository", (ALL_REPOSITORIES, *repositories), index=0)
filtered_ids = tuple(
    value
    for value in instance_ids
    if selected_repository == ALL_REPOSITORIES
    or repository_from_instance_id(value) == selected_repository
)
default_index = filtered_ids.index(DEFAULT_INSTANCE) if DEFAULT_INSTANCE in filtered_ids else 0
selected_id = st.sidebar.selectbox(
    "Demo instance",
    filtered_ids,
    index=default_index,
    format_func=lambda value: f"{value} — {repository_from_instance_id(value)}",
)
try:
    selected, adaptive_raw = selected_localization_view(selected_id)
    dataset_input = load_dataset_input(selected_id)
    with st.spinner("Preparing historical repository views..."):
        parsed, graph, file_documents, function_documents = prepare_repository_views(
            selected["checkout_path"],
            selected["repository"],
            selected["requested_commit"],
            selected["verified_commit"],
        )
except Exception as error:
    st.error(f"Unable to load the selected precomputed repository views: {error}")
    st.stop()

hybrid_rankings = selected["rankings"]["hybrid"]
adaptive_rankings = selected["rankings"]["adaptive"]
expansion_admissions = load_expansion_admissions(selected_id)

show_advanced_views = False

if not show_advanced_views:
    st.header("1. Dataset Input")
    st.write("Only localization-safe Loc-Bench fields are shown here.")
    st.json(
        {
            "instance_id": dataset_input["instance_id"],
            "repo": dataset_input["repo"],
            "base_commit": dataset_input["base_commit"],
            "problem_statement": dataset_input["problem_statement"],
            "category": dataset_input.get("category"),
        }
    )
    st.markdown(
        "**Gold fields withheld:** `patch`, `test_patch`, `labels`, "
        "`edit_functions`, `added_functions`"
    )
    st.info("Ground-truth modification locations remain hidden until localization is complete.")
    with st.expander("Full issue description", expanded=False):
        st.markdown(dataset_input["problem_statement"])
    with st.expander("Advanced Details"):
        st.write(
            "The local snapshot preserves the complete benchmark row. GraphFix separates "
            "issue-facing input from GraphFixGold so localization code cannot receive targets."
        )
        st.json(
            {
                "allowed_input_fields": [
                    "instance_id",
                    "repo",
                    "base_commit",
                    "problem_statement",
                    "category",
                ],
                "withheld_gold_fields": [
                    "patch",
                    "test_patch",
                    "labels",
                    "edit_functions",
                    "added_functions",
                ],
            },
            expanded=False,
        )

    st.header("2. Historical Repository Reconstruction")
    st.info(
        "`base_commit` is the benchmark-provided pre-fix code snapshot used for "
        "localization. GraphFix does not assume that it is exactly the commit from the "
        "issue-creation timestamp."
    )
    commits_match = (
        selected["requested_commit"].casefold() == selected["verified_commit"].casefold()
    )
    st.write(f"Repository: `{selected['repository']}`")
    st.code(
        f"Expected base_commit\n{selected['requested_commit']}\n"
        "        ↓\nHistorical repository reconstruction\n        ↓\n"
        f"Checked-out HEAD\n{selected['verified_commit']}"
    )
    if commits_match:
        st.success("✓ Exact commit verified")
    else:
        st.error("Checked-out HEAD does not match the requested historical commit.")
    st.write(
        "Python files parsed:",
        selected["parser_stats"]["python_files_parsed"],
    )
    with st.expander("Advanced Details"):
        st.write(f"Existing verified checkout: `{selected['checkout_path']}`")
        st.write(
            "Reconstruction uses cached Git objects and an isolated detached worktree. "
            "This walkthrough reads the existing checkout and does not run Git."
        )

    st.header("3. Python AST Parsing")
    parser_stats = selected["parser_stats"]
    parser_columns = st.columns(4)
    for column, (label, key) in zip(
        parser_columns,
        (
            ("Files scanned", "files_scanned"),
            ("Python files parsed", "python_files_parsed"),
            ("Parse failures", "parse_failures"),
            ("Classes", "classes"),
        ),
        strict=True,
    ):
        column.metric(label, parser_stats[key])
    parser_columns = st.columns(4)
    for column, (label, key) in zip(
        parser_columns,
        (
            ("Functions", "functions"),
            ("Methods", "methods"),
            ("Imports", "imports"),
            ("Calls", "calls"),
        ),
        strict=True,
    ):
        column.metric(label, parser_stats[key])

    parsed_paths = [item.file_path for item in parsed.files]
    preferred_vllm_path = "benchmarks/benchmark_serving.py"
    ranked_parsed_paths = [
        candidate["file_path"]
        for candidate in hybrid_rankings["file"]
        if candidate["file_path"] in parsed_paths
    ]
    if selected_id == DEFAULT_INSTANCE and preferred_vllm_path in parsed_paths:
        example_path = preferred_vllm_path
    elif ranked_parsed_paths:
        example_path = ranked_parsed_paths[0]
    else:
        example_path = parsed_paths[0]
    example_file = next(item for item in parsed.files if item.file_path == example_path)
    ranked_symbol_names = [
        candidate.get("qualified_name") or candidate.get("symbol_name")
        for candidate in hybrid_rankings["function"]
        if candidate["file_path"] == example_path
    ]
    example_symbol = next(
        (
            item
            for name in ranked_symbol_names
            for item in example_file.symbols
            if name in (item.name, item.qualified_name) or str(name).endswith(f".{item.name}")
        ),
        example_file.symbols[0] if example_file.symbols else None,
    )
    if example_symbol is not None:
        st.markdown("**Real parsed-code example**")
        st.write(f"File: `{example_symbol.file_path}`")
        st.write(f"Entity: `{example_symbol.qualified_name}`")
        st.write(f"Signature: `{example_symbol.signature or 'not available'}`")
        st.code(example_symbol.source[:1200], language="python")
    with st.expander("Advanced Details"):
        selected_file_path = st.selectbox(
            "Parsed Python file",
            parsed_paths,
            index=parsed_paths.index(example_path),
            key="guided_parsed_file",
        )
        parsed_file = next(item for item in parsed.files if item.file_path == selected_file_path)
        st.dataframe(
            [
                {
                    "Kind": symbol.kind,
                    "Name": symbol.name,
                    "Qualified name": symbol.qualified_name,
                    "Signature": symbol.signature,
                    "Lines": f"{symbol.start_line}-{symbol.end_line}",
                }
                for symbol in parsed_file.symbols
            ],
            width="stretch",
            hide_index=True,
        )
        st.json(parsed_file.model_dump(), expanded=False)

    st.header("4. Repository Graph")
    graph_stats = graph.stats
    graph_columns = st.columns(2)
    graph_columns[0].metric("Nodes", graph_stats.total_nodes)
    graph_columns[1].metric("Edges", graph_stats.total_edges)
    graph_seed = next(
        (
            candidate["document_id"]
            for candidate in hybrid_rankings["function"]
            if graph.get_node(candidate["document_id"]) is not None
        ),
        repository_root_id(graph),
    )
    simple_edges, simple_limited = neighborhood_edges(
        graph, graph_seed, 1, set(EDGE_TYPES), set(NODE_TYPES)
    )
    show_graph_sanity(graph, graph_seed, simple_edges)
    if simple_edges:
        st.graphviz_chart(graphviz_source(graph, simple_edges, graph_seed), width="stretch")
        if simple_limited:
            st.caption(f"Display capped at {MAX_GRAPH_EDGES} actual connected edges.")
    else:
        show_empty_graph_context(graph, graph_seed, set(EDGE_TYPES), 1)
    st.caption(
        "Light blue marks the selected real graph seed. Edge labels show actual relationships."
    )
    with st.expander("Advanced Graph Details"):
        advanced_seed = require_graph_root(
            graph,
            st.selectbox(
                "Graph seed node",
                [node.node_id for node in graph.nodes],
                index=next(
                    (index for index, node in enumerate(graph.nodes) if node.node_id == graph_seed),
                    0,
                ),
                key="guided_graph_seed",
            ),
        )
        graph_controls = st.columns(2)
        advanced_depth = graph_controls[0].radio(
            "Neighborhood depth", (1, 2), horizontal=True, key="guided_graph_depth"
        )
        advanced_types = set(
            graph_controls[1].multiselect(
                "Edge types", EDGE_TYPES, default=list(EDGE_TYPES), key="guided_graph_edge_types"
            )
        )
        advanced_edges, advanced_limited = neighborhood_edges(
            graph, advanced_seed, advanced_depth, advanced_types, set(NODE_TYPES)
        )
        if advanced_edges:
            st.graphviz_chart(
                graphviz_source(graph, advanced_edges, advanced_seed), width="stretch"
            )
            if advanced_limited:
                st.caption(f"Display capped at {MAX_GRAPH_EDGES} actual edges.")
        else:
            show_empty_graph_context(graph, advanced_seed, advanced_types, advanced_depth)
        show_node_inspector(
            parsed,
            graph,
            sorted(visible_node_ids(advanced_seed, advanced_edges)),
            key="guided_node_inspector",
        )

    st.header("5. Search Documents")
    st.write(
        "Parsed files and callable entities become stable retrieval-ready documents containing "
        "paths, symbols, signatures, relationships, docstrings, and bounded source."
    )
    file_id = hybrid_rankings["file"][0]["document_id"]
    function_id = hybrid_rankings["function"][0]["document_id"]
    file_document = next(
        (item for item in file_documents if item.document_id == file_id),
        file_documents[0],
    )
    function_document = next(
        (item for item in function_documents if item.document_id == function_id),
        function_documents[0],
    )
    st.markdown("**File SearchDocument**")
    st.json(
        {
            "document_id": file_document.document_id,
            "text_preview": file_document.text[:500],
        }
    )
    st.markdown("**Function SearchDocument**")
    st.json(
        {
            "document_id": function_document.document_id,
            "text_preview": function_document.text[:500],
        }
    )
    with st.expander("Advanced Details"):
        document_level = st.radio(
            "Document level",
            ("File", "Function / method"),
            horizontal=True,
            key="guided_document_level",
        )
        documents = file_documents if document_level == "File" else function_documents
        document = st.selectbox(
            "Complete SearchDocument", documents, format_func=document_label, key="guided_document"
        )
        st.json(document.model_dump(), expanded=False)
        st.code(document.text)
        st.write("BM25 token preview:", list(tokenize(document.text)[:150]))

    st.header("6. Candidate Retrieval")
    st.write(
        "BM25 uses lexical matching. Dense retrieval uses semantic similarity. "
        "Hybrid combines both rankings using Reciprocal Rank Fusion (RRF)."
    )
    for method, label in (("bm25", "BM25"), ("dense", "Dense"), ("hybrid", "Hybrid")):
        st.markdown(f"**{label} — file Top 5**")
        st.dataframe(
            ranking_table(selected["rankings"][method]["file"][:5]),
            width="stretch",
            hide_index=True,
        )
    st.caption(
        "Hybrid combines the complete BM25 and Dense rankings with Reciprocal Rank Fusion "
        "before Top-10 truncation. Gold relevance is not shown here."
    )
    with st.expander("Advanced Details"):
        retrieval_level = st.radio(
            "Localization level",
            ("file", "function"),
            horizontal=True,
            key="guided_retrieval_level",
        )
        for method, label in (("bm25", "BM25"), ("dense", "Dense"), ("hybrid", "Hybrid")):
            st.markdown(f"**{label} Top 10**")
            st.dataframe(
                ranking_table(selected["rankings"][method][retrieval_level]),
                width="stretch",
                hide_index=True,
            )

    st.header("7. Static Graph Baseline")
    st.code(
        "Hybrid candidates\n"
        "        ↓\n"
        "Depth-1 CALLS / IMPORTS / INHERITS expansion\n"
        "        ↓\n"
        "Structural reranking"
    )
    st.write("Static expansion adds useful structural neighbors, but can also introduce noise.")
    static_seed = hybrid_rankings["function"][0]["document_id"]
    static_edges, _ = neighborhood_edges(
        graph, static_seed, 1, {"CALLS", "IMPORTS", "INHERITS"}, set(NODE_TYPES)
    )
    if static_edges:
        st.graphviz_chart(graphviz_source(graph, static_edges, static_seed), width="stretch")
    else:
        show_empty_graph_context(graph, static_seed, {"CALLS", "IMPORTS", "INHERITS"}, 1)
    st.dataframe(
        ranking_table(selected["rankings"]["graph"]["file"][:5]),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Advanced Details"):
        static_level = st.radio(
            "Static Graph level", ("file", "function"), horizontal=True, key="guided_static_level"
        )
        st.dataframe(
            ranking_table(selected["rankings"]["graph"][static_level]),
            width="stretch",
            hide_index=True,
        )

    st.header("8. Adaptive Graph Navigation")
    st.warning("Saved pilot trace only — this walkthrough does not call Ollama.")
    file_trace = selected["file_navigation"]["trace"]
    st.dataframe(trace_table(file_trace[:5]), width="stretch", hide_index=True)
    raw_navigation = adaptive_raw["file_navigation"]
    step_number = 1
    with st.expander("Advanced Trace Details"):
        step_number = st.select_slider(
            "Saved adaptive step",
            options=range(1, len(raw_navigation["trace"]) + 1),
            key="guided_trace_step",
        )
        trace_level = st.radio(
            "Trace level", ("file", "function"), horizontal=True, key="guided_trace_level"
        )
        raw_level = adaptive_raw[f"{trace_level}_navigation"]
        st.markdown("**Stop reason**")
        st.write(stop_reason_text(raw_level["stopping_reason"]))
        st.dataframe(
            trace_table(selected[f"{trace_level}_navigation"]["trace"]),
            width="stretch",
            hide_index=True,
        )
        detail_index = st.selectbox(
            "Trace row details",
            range(len(raw_level["trace"])),
            format_func=lambda value: (
                f"Step {value + 1}: {raw_level['trace'][value]['action']} — "
                f"{readable_node_id(raw_level['trace'][value]['node_id'])}"
            ),
            key=f"guided_trace_detail_{trace_level}",
        )
        detail = raw_level["trace"][detail_index]
        previously_seen = {
            node_id
            for prior_step in raw_level["trace"][:detail_index]
            for node_id in prior_step["observed_node_ids"]
        }
        returned = set(detail["observed_node_ids"])
        st.markdown("**Complete target identifier**")
        st.code(detail["node_id"] or "STOP")
        st.markdown("**Complete returned observation**")
        st.write(detail["message"])
        st.write(f"Returned nodes: {len(returned)} · New nodes: {len(returned - previously_seen)}")
        if detail["validation_errors"]:
            st.markdown("**Rejected controller-response details**")
            for validation_error in detail["validation_errors"]:
                st.write(validation_error)
        st.caption(
            "Returned nodes are all nodes in this observation; new nodes exclude "
            "nodes already seen in earlier steps."
        )
        st.json(raw_level, expanded=False)
    raw_step = raw_navigation["trace"][step_number - 1]
    prior = {
        node_id
        for step in raw_navigation["trace"][: step_number - 1]
        for node_id in step["observed_node_ids"]
    }
    newly_observed = set(raw_step["observed_node_ids"]) - prior
    raw_target = raw_step["node_id"]
    target = (
        require_graph_root(graph, raw_target)
        if isinstance(raw_target, str)
        else stop_fallback_target(graph, raw_navigation["trace"], step_number - 1)
    )
    step_edges, step_limited = adaptive_connected_edges(graph, target, prior, newly_observed)
    if step_edges:
        st.graphviz_chart(
            graphviz_source(
                graph,
                step_edges,
                target,
                previous=prior,
                new=newly_observed,
                seed_color="#f4a261",
            ),
            width="stretch",
        )
        if step_limited:
            st.caption(f"Display capped at {MAX_GRAPH_EDGES} actual edges.")
    else:
        show_empty_graph_context(graph, target, set(EDGE_TYPES), 1)
    st.caption(
        "Orange = selected target; green = newly observed evidence; gray = previously observed."
    )

    st.header("9. Final Localization")
    st.write("Observed graph evidence reranks the original Hybrid Top-10.")
    with st.container(border=True):
        st.markdown("### Selected-Issue Localization Result")
        for level, label in (("file", "File"), ("function", "Callable")):
            hybrid_level = hybrid_rankings[level]
            adaptive_level = adaptive_rankings[level]
            hybrid_top = hybrid_level[0]
            adaptive_top = adaptive_level[0]
            hybrid_map = candidate_rank_map(hybrid_level)
            adaptive_map = candidate_rank_map(adaptive_level)
            promoted = sum(
                adaptive_map.get(candidate_id, rank) < rank
                for candidate_id, rank in hybrid_map.items()
            )
            reordered = sum(
                adaptive_map.get(candidate_id, rank) != rank
                for candidate_id, rank in hybrid_map.items()
            )
            columns = st.columns(2)
            for column, method, candidate in (
                (columns[0], "Hybrid", hybrid_top),
                (columns[1], "Adaptive Fixed", adaptive_top),
            ):
                with column.container(border=True):
                    st.markdown(f"**{label} — {method} Top-1**")
                    if level == "function":
                        st.write(
                            candidate.get("qualified_name")
                            or candidate.get("symbol_name")
                            or "Unnamed callable"
                        )
                        st.caption(f"File: {candidate['file_path']}")
                    else:
                        st.write(candidate["file_path"])
            st.caption(f"{label} candidates promoted: {promoted}; reordered: {reordered}.")
        st.caption("Gold locations remain hidden here.")
    file_comparison = [
        {
            "Hybrid rank": candidate["rank"],
            "Hybrid path": candidate["file_path"],
            "Adaptive rank": candidate_rank_map(adaptive_rankings["file"]).get(
                candidate["document_id"], "—"
            ),
        }
        for candidate in hybrid_rankings["file"]
    ]
    function_comparison = [
        {
            "Hybrid rank": candidate["rank"],
            "Hybrid candidate": candidate["qualified_name"],
            "Adaptive rank": candidate_rank_map(adaptive_rankings["function"]).get(
                candidate["document_id"], "—"
            ),
        }
        for candidate in hybrid_rankings["function"]
    ]
    st.markdown("**Files: Hybrid ranking → Adaptive ranking**")
    st.dataframe(
        file_comparison[:5],
        width="stretch",
        hide_index=True,
    )
    st.markdown("**Functions/methods: Hybrid ranking → Adaptive ranking**")
    st.dataframe(
        function_comparison[:5],
        width="stretch",
        hide_index=True,
    )
    st.info("The saved fixed-pool baseline reorders only the original Hybrid Top-10.")
    if expansion_admissions is not None:
        admitted = [
            *expansion_admissions["expanded_file_candidate_ids"],
            *expansion_admissions["expanded_function_candidate_ids"],
        ]
        with st.expander("Advanced — Candidate-expansion ablation", expanded=False):
            if admitted:
                st.write("Observed outside-pool candidates admitted:")
                st.write(admitted)
            else:
                st.write("No outside-pool candidate was admitted for this instance.")
    with st.expander("Advanced Details"):
        st.markdown("**Complete file ranking comparison**")
        st.dataframe(file_comparison, width="stretch", hide_index=True)
        st.markdown("**Complete function/method ranking comparison**")
        st.dataframe(function_comparison, width="stretch", hide_index=True)

    st.divider()
    st.success("LOCALIZATION COMPLETE")
    st.markdown("### Gold data has not been used above this point.")
    st.divider()
    st.subheader("POST-HOC GOLD EVALUATION")

    st.header("10. Post-Hoc Gold Evaluation")
    gold_model = load_dataset_gold(selected_id)
    gold = extract_gold_locations(gold_model)
    hybrid_metrics = evaluate_rankings(
        ranked_candidates(hybrid_rankings["file"]),
        ranked_candidates(hybrid_rankings["function"]),
        gold,
    )
    adaptive_metrics = evaluate_rankings(
        ranked_candidates(adaptive_rankings["file"]),
        ranked_candidates(adaptive_rankings["function"]),
        gold,
    )
    expansion = load_expansion_details().get(selected_id)
    st.markdown("### Selected-Issue Result")
    st.write("**Gold files**", list(gold.files) or "No file targets recorded")
    st.write("**Gold functions**", list(gold.functions) or "No function targets recorded")
    rank_columns = st.columns(2)
    rank_columns[0].metric(
        "Relevant file rank: Hybrid → Adaptive",
        f"{rank_text(hybrid_metrics.file.first_relevant_rank)} → "
        f"{rank_text(adaptive_metrics.file.first_relevant_rank)}",
    )
    rank_columns[1].metric(
        "Relevant function rank: Hybrid → Adaptive",
        (
            f"{rank_text(hybrid_metrics.function.first_relevant_rank)} → "
            f"{rank_text(adaptive_metrics.function.first_relevant_rank)}"
        ),
    )
    metric_rows = []
    for method, metrics in (("Hybrid", hybrid_metrics), ("Adaptive Fixed", adaptive_metrics)):
        for level in ("file", "function"):
            values = metrics.file if level == "file" else metrics.function
            metric_rows.append(
                {
                    "Method": method,
                    "Level": level.title(),
                    "Hit@1": values.hit_at_1,
                    "Recall@5": values.recall_at_5,
                    "Recall@10": values.recall_at_10,
                    "MRR": values.mrr,
                }
            )
    st.dataframe(metric_rows, width="stretch", hide_index=True)
    gold_evidence = relevant_evidence(adaptive_raw, set(gold.files), set(gold.functions))
    if gold_evidence:
        st.markdown("**Gold-relevant evidence discovered during navigation**")
        st.dataframe(
            [
                {
                    "Discovered node": node["node_id"],
                    "Confirmed gold levels": ", ".join(node["gold_levels"]),
                }
                for node in gold_evidence
            ],
            width="stretch",
            hide_index=True,
        )
        st.write(
            "During graph navigation, this function was discovered and was later confirmed "
            "to be gold-relevant."
        )
    if expansion is not None:
        st.markdown("**Fixed pool → Expanded pool (same saved navigation trace)**")
        st.write(
            "File rank:",
            f"{rank_text(expansion['fixed_file_rank'])} → "
            f"{rank_text(expansion['expanded_file_rank'])}",
        )
        st.write(
            "Function rank:",
            f"{rank_text(expansion['fixed_function_rank'])} → "
            f"{rank_text(expansion['expanded_function_rank'])}",
        )
        st.warning(
            "Bounded expansion now permits observed outside-pool candidates to enter, "
            "but it improved 0 of 40 paired pilot outcomes and degraded 2. Fixed pool "
            "therefore remains the default."
        )
    with st.expander("Advanced Evaluation Details"):
        st.json(
            {
                "labels": gold_model.labels,
                "edit_functions": gold_model.edit_functions,
                "added_functions": gold_model.added_functions,
                "hybrid_file_metrics": hybrid_metrics.file.model_dump(),
                "adaptive_file_metrics": adaptive_metrics.file.model_dump(),
                "hybrid_function_metrics": hybrid_metrics.function.model_dump(),
                "adaptive_function_metrics": adaptive_metrics.function.model_dump(),
                "candidate_expansion": expansion,
                "patch": gold_model.patch,
                "test_patch": gold_model.test_patch,
            },
            expanded=False,
        )

    st.subheader("What this example shows")
    st.info(
        rank_takeaway(
            "file",
            hybrid_metrics.file.first_relevant_rank,
            adaptive_metrics.file.first_relevant_rank,
        )
        + " "
        + rank_takeaway(
            "function",
            hybrid_metrics.function.first_relevant_rank,
            adaptive_metrics.function.first_relevant_rank,
        )
    )
    with st.expander("Advanced Details: full evaluation context"):
        st.write(
            "The table below is the full 560-instance deterministic baseline comparison. "
            "The definitive Adaptive Fixed cohort result is summarized separately below."
        )
        st.dataframe(metric_table(full_rows), width="stretch", hide_index=True)
        st.json(load_json(FULL560_SUMMARY), expanded=False)
