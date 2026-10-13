"""Polished Streamlit presentation for fresh public-GitHub localization."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import streamlit as st

from graphfix.agent import OllamaConfig, OllamaTextGenerator
from graphfix.live import (
    LiveLocalizationService,
    LiveMethodResult,
    PreparedLiveRepository,
    live_input,
    parse_public_github_url,
    resolve_default_branch_head,
    save_live_artifact,
    validate_commit,
)
from graphfix.pipeline import AdaptiveNavigationPipeline
from graphfix.pipeline.advanced import AdvancedLocalizationPipeline
from graphfix.pipeline.models import AdaptiveLocalizationResult, AdaptiveNavigationConfig
from graphfix.repositories import RepositoryManager
from graphfix.retrieval import EmbeddingCache, RankedCandidate, SentenceTransformerEncoder

ROOT = Path(__file__).resolve().parents[1]
LIVE_CACHE = ROOT / "data" / "cache" / "live_repositories"
LIVE_RESULTS = ROOT / "results" / "live_repository"
LIVE_EMBEDDINGS = ROOT / ".cache" / "graphfix" / "live_embeddings"
EXAMPLE_NAME = "Budget-Aware AutoLLM — Scientific notation"
CUSTOM_NAME = "Custom repository"
PRESET_URL = "https://github.com/UpadhyayKiran/Budget-Aware-AutoLLM"
PRESET_TITLE = "Support scientific notation in numeric answer extraction"
PRESET_BODY = (
    "Numeric answer evaluation incorrectly handles scientific notation. A response "
    "containing ‘Final answer: 1e3’ is interpreted as 1 rather than 1000, and ‘Final "
    "answer: 2.5e-2’ is interpreted as 2.5 rather than 0.025. Support positive and "
    "negative exponents while preserving existing decimal, fraction, percentage and "
    "marked-answer precedence behaviour."
)


def _change_example() -> None:
    """Populate editable inputs without starting localization."""
    if st.session_state.live_example == EXAMPLE_NAME:
        st.session_state.live_url = PRESET_URL
        st.session_state.live_title = PRESET_TITLE
        st.session_state.live_body = PRESET_BODY
    else:
        st.session_state.live_url = ""
        st.session_state.live_title = ""
        st.session_state.live_body = ""
    st.session_state.live_commit = ""


@st.cache_data(show_spinner=False)
def _dense_availability() -> tuple[bool, str]:
    """Check local assets without loading model weights or downloading."""
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
        )
        return True, f"MiniLM is available locally ({path})."
    except Exception as error:
        return False, f"MiniLM is unavailable locally: {type(error).__name__}: {error}"


@st.cache_resource(show_spinner=False)
def _local_encoder() -> tuple[SentenceTransformerEncoder | None, str]:
    available, status = _dense_availability()
    if not available:
        return None, status
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
        )
        return SentenceTransformerEncoder(path), status
    except Exception as error:
        return None, f"MiniLM could not be loaded: {type(error).__name__}: {error}"


@st.cache_data(ttl=5, show_spinner=False)
def _ollama_status(endpoint: str, model: str) -> tuple[bool, str]:
    request = urllib.request.Request(f"{endpoint.rstrip('/')}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        return False, f"Ollama is unavailable at {endpoint}: {error}"
    names = {
        item.get("name")
        for item in payload.get("models", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if model not in names:
        return False, f"Ollama is reachable, but {model!r} is not installed."
    return True, f"Ollama and {model!r} are available."


def _ranking_rows(rankings: tuple[RankedCandidate, ...]) -> list[dict[str, Any]]:
    rows = []
    for item in rankings:
        row: dict[str, Any] = {
            "Rank": item.rank,
            "File": item.file_path,
            "Score": round(item.score, 6),
        }
        if item.kind != "file":
            row.update(
                {
                    "Callable": item.qualified_name or item.symbol_name,
                    "Lines": (
                        f"{item.start_line}–{item.end_line}"
                        if item.start_line is not None and item.end_line is not None
                        else "—"
                    ),
                }
            )
        rows.append(row)
    return rows


def _readable_node(node_id: str | None) -> str:
    """Return a concise trace label without altering the stored node identifier."""
    if not node_id:
        return "STOP"
    parts = node_id.split(":", 3)
    node_type = parts[0] if parts else "node"
    qualified = parts[2] if len(parts) > 2 else node_id
    name = qualified.rsplit(".", 1)[-1]
    suffix = "()" if node_type in {"function", "method"} else ""
    return f"{name}{suffix}"


def _stop_reason_text(reason: str) -> str:
    return {
        "explicit_stop": "Controller selected STOP",
        "invalid_response_stop": "Retry limit reached after invalid controller responses",
        "max_steps": "Maximum navigation-step budget reached",
    }.get(reason, reason.replace("_", " ").title())


def _source_snippet(prepared: PreparedLiveRepository, candidate: RankedCandidate) -> str:
    source_path = (prepared.checkout_path / candidate.file_path).resolve()
    if (
        not source_path.is_relative_to(prepared.checkout_path.resolve())
        or not source_path.is_file()
    ):
        return "Source is unavailable for this candidate."
    source = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if candidate.start_line is None:
        start, end = 0, min(len(source), 120)
    else:
        start = max(0, candidate.start_line - 4)
        end = min(len(source), (candidate.end_line or candidate.start_line) + 3)
    return "\n".join(f"{index + 1:>5}  {source[index]}" for index in range(start, end))


def _neighborhood(prepared: PreparedLiveRepository, seed: str) -> tuple[list[Any], set[str]]:
    edges = [edge for edge in prepared.graph.edges if seed in {edge.source, edge.target}][:30]
    nodes = {seed}
    for edge in edges:
        nodes.update((edge.source, edge.target))
    return edges, nodes


def _graphviz(prepared: PreparedLiveRepository, seed: str) -> str:
    edges, node_ids = _neighborhood(prepared, seed)
    nodes = {node.node_id: node for node in prepared.graph.nodes if node.node_id in node_ids}
    lines = [
        "digraph GraphFixLiveRepository {",
        "rankdir=LR;",
        'graph [pad="0.2", nodesep="0.35", ranksep="0.55"];',
        'node [shape=box, style="rounded", fontsize="10"];',
        'edge [fontsize="9", color="#66788a"];',
    ]
    for node_id in sorted(node_ids):
        node = nodes.get(node_id)
        label = f"{node.node_type}: {node.name}" if node is not None else node_id
        attributes = f"label={json.dumps(label)}"
        if node_id == seed:
            attributes += ', style="rounded,filled", fillcolor="#b8d8f0"'
        lines.append(f"{json.dumps(node_id)} [{attributes}];")
    for edge in edges:
        lines.append(
            f"{json.dumps(edge.source)} -> {json.dumps(edge.target)} "
            f"[label={json.dumps(edge.edge_type)}];"
        )
    lines.append("}")
    return "\n".join(lines)


def _run_adaptive(
    prepared: PreparedLiveRepository,
    encoder: SentenceTransformerEncoder,
    endpoint: str,
    model: str,
) -> AdaptiveLocalizationResult:
    manager = RepositoryManager(LIVE_CACHE)
    advanced = AdvancedLocalizationPipeline(
        manager, encoder=encoder, embedding_cache=EmbeddingCache(LIVE_EMBEDDINGS)
    )
    config = AdaptiveNavigationConfig(model=model, endpoint=endpoint, candidate_pool="fixed")
    return AdaptiveNavigationPipeline(
        manager,
        OllamaTextGenerator(
            OllamaConfig(model=model, endpoint=endpoint, timeout_seconds=config.timeout_seconds)
        ),
        config=config,
        hybrid_pipeline=advanced,
    ).localize(prepared.issue)


def _saved_payload(url: str, title: str, body: str) -> tuple[Path, dict[str, Any]] | None:
    try:
        repository = parse_public_github_url(url)
    except ValueError:
        return None
    expected = "\n\n".join(part.strip() for part in (title, body) if part.strip())
    for path in sorted(LIVE_RESULTS.glob("live-*/localization.json"), reverse=True):
        try:
            payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
            issue = cast(dict[str, Any], payload.get("issue", {}))
            if issue.get("repo") == repository and issue.get("problem_statement") == expected:
                return path, payload
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return None


def _load_saved_run(payload_path: Path, payload: dict[str, Any]) -> tuple[Any, ...]:
    """Load rankings and rebuild views only; never rerun retrieval or inference."""
    started = perf_counter()
    issue_payload = cast(dict[str, Any], payload["issue"])
    issue = live_input(
        cast(str, payload["repository"]),
        cast(str, payload["resolved_commit"]),
        "",
        cast(str, issue_payload["problem_statement"]),
    )
    service = LiveLocalizationService(LIVE_CACHE, embedding_cache=EmbeddingCache(LIVE_EMBEDDINGS))
    prepared = service.prepare(issue)
    methods_payload = cast(dict[str, dict[str, Any]], payload["methods"])
    computed = {
        name: LiveMethodResult(
            method=name,
            file_rankings=tuple(
                RankedCandidate.model_validate_json(json.dumps(item))
                for item in details["file_rankings"]
            ),
            function_rankings=tuple(
                RankedCandidate.model_validate_json(json.dumps(item))
                for item in details["function_rankings"]
            ),
            elapsed_seconds=float(details["elapsed_seconds"]),
        )
        for name, details in methods_payload.items()
    }
    adaptive_payload = payload.get("adaptive")
    adaptive = (
        AdaptiveLocalizationResult.model_validate_json(json.dumps(adaptive_payload))
        if isinstance(adaptive_payload, dict)
        else None
    )
    return (
        prepared,
        computed,
        adaptive,
        payload_path,
        cast(str, payload.get("default_branch", "saved")),
        "Saved run loaded",
        perf_counter() - started,
    )


def _stage(number: int, title: str, explanation: str) -> None:
    st.header(f"{number}. {title}")
    st.write(explanation)


def _show_rankings(prepared: PreparedLiveRepository, name: str, result: LiveMethodResult) -> None:
    descriptions = {
        "bm25": "Lexical relevance over AST-derived SearchDocuments.",
        "dense": "MiniLM cosine similarity over the same documents.",
        "hybrid": "Reciprocal Rank Fusion of complete BM25 and Dense rankings.",
        "graph": "Hybrid fused with bounded static RepositoryGraph evidence.",
    }
    st.markdown(f"#### {name.replace('_', ' ').title()}")
    st.caption(f"{descriptions.get(name, '')} Completed in {result.elapsed_seconds:.2f}s.")
    file_tab, function_tab = st.tabs(["Candidate files", "Candidate functions"])
    with file_tab:
        st.dataframe(
            _ranking_rows(result.file_rankings),
            width="stretch",
            hide_index=True,
            height=390,
        )
    with function_tab:
        st.dataframe(
            _ranking_rows(result.function_rankings),
            width="stretch",
            hide_index=True,
            height=390,
        )
    with st.expander("Inspect candidate source", expanded=False):
        candidates = (*result.file_rankings, *result.function_rankings)
        selected_id = st.selectbox(
            "Candidate",
            [item.document_id for item in candidates],
            format_func=lambda value: next(
                (
                    item.qualified_name or item.file_path
                    for item in candidates
                    if item.document_id == value
                ),
                value,
            ),
            key=f"live_source_{name}",
        )
        selected = next(item for item in candidates if item.document_id == selected_id)
        st.caption(f"Complete selection: `{selected.document_id}`")
        st.caption(
            f"`{selected.file_path}` · lines {selected.start_line or 1}–{selected.end_line or '…'}"
        )
        st.code(_source_snippet(prepared, selected), language="python")


def _render_results(run: tuple[Any, ...]) -> None:
    prepared, computed, adaptive, artifact, branch, origin, total_elapsed = run
    st.divider()
    st.success("Localization complete")
    st.subheader("Run overview")
    overview = st.columns(3)
    overview[0].markdown("**Repository**")
    overview[0].write(prepared.issue.repo)
    overview[1].metric("Resolved commit", prepared.issue.base_commit[:12])
    overview[2].metric("Run time", f"{total_elapsed:.2f}s")
    structure = st.columns(3)
    structure[0].metric("Python files", prepared.parsed.stats.python_files_parsed)
    structure[1].metric("Graph nodes", prepared.graph.stats.total_nodes)
    structure[2].metric("Graph edges", prepared.graph.stats.total_edges)
    _stage(1, "Issue and repository", "The issue and repository identity used for this run.")
    with st.container(border=True):
        parts = prepared.issue.problem_statement.split("\n\n", 1)
        st.markdown(f"**{parts[0]}**")
        st.caption(prepared.issue.repo)
        if len(parts) > 1:
            preview = parts[1] if len(parts[1]) <= 600 else parts[1][:600].rstrip() + "…"
            st.write(preview)
    with st.expander("Full issue description", expanded=False):
        st.markdown(prepared.issue.problem_statement)
    _stage(2, "Source snapshot", "GraphFix localizes against one exact, immutable commit.")
    with st.container(border=True):
        st.markdown(f"**Repository:** `{prepared.issue.repo}`")
        st.markdown(f"**Pinned source:** `{branch}` at `{prepared.issue.base_commit[:12]}`")
        st.caption("The complete SHA and local provenance are available below.")
    with st.expander("Complete source provenance", expanded=False):
        st.code(prepared.issue.base_commit)
        st.json(
            {
                "default_branch": branch,
                "checkout_cache_reused": prepared.cache_reused,
                "checkout_path": str(prepared.checkout_path),
                "saved_artifact": str(artifact),
            },
            expanded=False,
        )
    _stage(3, "Python structure", "Source is parsed without executing repository code.")
    parser = st.columns(4)
    parser[0].metric("Python files", prepared.parsed.stats.python_files_parsed)
    parser[1].metric("Classes", prepared.parsed.stats.classes)
    parser[2].metric("Callables", prepared.parsed.stats.functions + prepared.parsed.stats.methods)
    parser[3].metric("Parse failures", prepared.parsed.stats.parse_failures)
    with st.expander("Parser details", expanded=False):
        st.json(prepared.parsed.stats.model_dump(mode="json"), expanded=False)
    _stage(
        4, "Repository graph", "The complete parsed repository supplies structural relationships."
    )
    graph_metrics = st.columns(3)
    graph_metrics[0].metric("Graph nodes", prepared.graph.stats.total_nodes)
    graph_metrics[1].metric("Graph edges", prepared.graph.stats.total_edges)
    graph_metrics[2].metric(
        "Relationship types", len({edge.edge_type for edge in prepared.graph.edges})
    )
    graph_seed = next(
        (
            item.document_id
            for method in ("hybrid", "bm25", "dense", "graph")
            for item in computed.get(method, LiveMethodResult("", (), (), 0)).function_rankings
        ),
        None,
    )
    if graph_seed is not None:
        with st.expander("Explore local graph neighborhood", expanded=True):
            candidates = [
                node.node_id
                for node in prepared.graph.nodes
                if node.node_type in {"function", "method", "class", "file"}
            ]
            seed = st.selectbox(
                "Graph symbol",
                candidates,
                index=candidates.index(graph_seed),
                format_func=_readable_node,
                key="live_graph_seed",
            )
            edges, nodes = _neighborhood(prepared, seed)
            facts = st.columns(3)
            selected_node = prepared.graph.get_node(seed)
            facts[0].markdown("**Selected node type**")
            facts[0].write(selected_node.node_type if selected_node else "—")
            facts[1].metric("Visible nodes", len(nodes))
            facts[2].metric("Visible edges", len(edges))
            st.caption(f"Complete selection: `{selected_node.file_path if selected_node else '—'}`")
            with st.expander("Complete graph-node identifier", expanded=False):
                st.code(seed)
            st.graphviz_chart(_graphviz(prepared, seed), width="stretch")
    _stage(
        5,
        "Search documents",
        "File and callable documents are derived independently from the parsed repository.",
    )
    docs = st.columns(2)
    docs[0].metric("File documents", len(prepared.file_documents))
    docs[1].metric("Callable documents", len(prepared.function_documents))
    with st.expander("Inspect a SearchDocument", expanded=False):
        available = (*prepared.file_documents, *prepared.function_documents)
        document_id = st.selectbox(
            "Search document",
            [item.document_id for item in available],
            format_func=lambda value: next(
                (
                    item.qualified_name or item.file_path
                    for item in available
                    if item.document_id == value
                ),
                value,
            ),
            key="live_document",
        )
        document = next(item for item in available if item.document_id == document_id)
        st.caption(f"Complete selection: `{document.document_id}`")
        st.json(document.model_dump(exclude={"text"}, mode="json"), expanded=False)
        st.code(document.text[:4000], language="text")
    retrieval = {name: result for name, result in computed.items() if name != "graph"}
    if retrieval:
        _stage(6, "Retrieval results", "Choose a completed method to inspect its actual rankings.")
        selected_method = st.selectbox("Retrieval method", list(retrieval), format_func=str.upper)
        _show_rankings(prepared, selected_method, retrieval[selected_method])
    if "graph" in computed:
        _stage(
            7,
            "Static graph expansion",
            "This separate baseline fuses Hybrid with bounded structural evidence.",
        )
        _show_rankings(prepared, "graph", computed["graph"])
    if adaptive is not None:
        _stage(
            8,
            "Adaptive navigation",
            "A prompted controller selects bounded actions; deterministic code validates, "
            "executes, and reranks.",
        )
        st.warning("Adaptive navigation is not guaranteed to improve every issue ranking.")
        for level in (adaptive.file_navigation, adaptive.function_navigation):
            with st.expander(
                f"{level.localization_level.title()} navigation trace", expanded=False
            ):
                stats = st.columns(3)
                stats[0].metric("Steps", level.actual_steps)
                stats[1].metric("Model calls", level.model_calls)
                stats[2].metric("Rejected controller responses", level.validation_failures)
                st.markdown("**Stop reason**")
                st.write(_stop_reason_text(level.stopping_reason))
                seen: set[str] = set()
                trace_rows = []
                new_counts: list[int] = []
                for index, step in enumerate(level.trace, 1):
                    returned = set(step.observed_node_ids)
                    newly_discovered = returned - seen
                    seen.update(returned)
                    new_counts.append(len(newly_discovered))
                    trace_rows.append(
                        {
                            "Step": index,
                            "Action": step.action,
                            "Target": _readable_node(step.node_id),
                            "Returned nodes": len(returned),
                            "New nodes": len(newly_discovered),
                            "Rejected responses": len(step.invalid_attempts),
                            "Observation": step.message,
                        }
                    )
                st.dataframe(
                    trace_rows,
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Returned nodes are all nodes in an observation; new nodes exclude "
                    "nodes seen earlier. Rejected controller responses count invalid model "
                    "attempts. Validation-error messages describe those same attempts and "
                    "are not added again."
                )
                detail_options = list(range(len(level.trace)))

                def trace_detail_label(value: int, trace: Any = level.trace) -> str:
                    trace_step = trace[value]
                    return (
                        f"Step {value + 1}: {trace_step.action} — "
                        f"{_readable_node(trace_step.node_id)}"
                    )

                detail_index = st.selectbox(
                    "Trace row details",
                    detail_options,
                    format_func=trace_detail_label,
                    key=f"live_trace_detail_{level.localization_level}",
                )
                detail = level.trace[detail_index]
                st.markdown("**Complete target identifier**")
                st.code(detail.node_id or "STOP")
                st.markdown("**Complete returned observation**")
                st.write(detail.message)
                st.write(
                    f"Returned nodes: {len(detail.observed_node_ids)} · "
                    f"New nodes: {new_counts[detail_index]}"
                )
                if detail.observed_nodes:
                    st.dataframe(
                        [
                            {
                                "Name": node.name,
                                "File": node.file_path,
                                "Relationship": node.relationship,
                                "Direction": node.direction,
                            }
                            for node in detail.observed_nodes
                        ],
                        width="stretch",
                        hide_index=True,
                    )
                if detail.invalid_attempts:
                    st.markdown("**Rejected controller-response details**")
                    for attempt in detail.invalid_attempts:
                        st.write(f"Attempt {attempt.attempt_number}: {attempt.validation_error}")
    final_method = (
        "adaptive"
        if adaptive is not None
        else next((name for name in ("graph", "hybrid", "bm25", "dense") if name in computed), None)
    )
    if final_method is None:
        return
    _stage(
        9,
        "Final candidate files and functions",
        "Localization candidates—not correctness or benchmark metrics.",
    )
    final_files = (
        adaptive.file_rankings if adaptive is not None else computed[final_method].file_rankings
    )
    final_functions = (
        adaptive.function_rankings
        if adaptive is not None
        else computed[final_method].function_rankings
    )
    st.caption(f"Final displayed method: {final_method.replace('_', ' ').title()}")
    file_tab, function_tab = st.tabs(["Top candidate files", "Top candidate functions"])
    with file_tab:
        st.dataframe(_ranking_rows(final_files), width="stretch", hide_index=True)
    with function_tab:
        st.dataframe(_ranking_rows(final_functions), width="stretch", hide_index=True)
    with st.expander("Complete run configuration and saved output", expanded=False):
        st.json(
            {
                "mode": "live_public_github",
                "completed_methods": list(computed),
                "adaptive_completed": adaptive is not None,
                "artifact": str(artifact),
            },
            expanded=False,
        )


def render_live_repository_tab() -> None:
    """Render live mode; computation occurs only after an explicit button."""
    st.title("GraphFix — Live Repository Localization")
    st.caption("Fresh issue localization for public GitHub repositories with Python source")
    with st.expander("About this mode", expanded=False):
        st.write(
            "GraphFix pins an exact source snapshot, parses supported Python files, and "
            "ranks candidate files and callables. It does not execute repository code, "
            "install target dependencies, or report benchmark metrics for an unlabeled issue."
        )
    st.session_state.setdefault("live_example", EXAMPLE_NAME)
    st.session_state.setdefault("live_url", PRESET_URL)
    st.session_state.setdefault("live_title", PRESET_TITLE)
    st.session_state.setdefault("live_body", PRESET_BODY)
    st.session_state.setdefault("live_commit", "")
    st.selectbox(
        "Choose an example",
        [EXAMPLE_NAME, CUSTOM_NAME],
        key="live_example",
        on_change=_change_example,
    )
    with st.container(border=True):
        url = st.text_input(
            "GitHub repository", key="live_url", placeholder="https://github.com/owner/repository"
        )
        title = st.text_input("Issue title", key="live_title")
        body = st.text_area("Issue description", key="live_body", height=150)
    with st.expander("Advanced settings", expanded=False):
        commit = st.text_input(
            "Commit SHA",
            key="live_commit",
            help="Leave blank to use the latest default-branch commit.",
        )
        methods = st.multiselect(
            "Localization methods",
            ["BM25", "Dense + Hybrid + Static Graph", "Adaptive Fixed"],
            default=["BM25", "Dense + Hybrid + Static Graph"],
            key="live_methods",
        )
        endpoint = st.text_input(
            "Ollama endpoint", "http://127.0.0.1:11434", key="live_ollama_endpoint"
        )
        model = st.text_input("Ollama model", "qwen2.5-coder:7b", key="live_ollama_model")
        dense_ok, dense_status = _dense_availability()
        (st.success if dense_ok else st.warning)(
            "MiniLM ready" if dense_ok else "MiniLM unavailable"
        )
        adaptive_ok, adaptive_status = _ollama_status(endpoint, model)
        (st.success if adaptive_ok else st.warning)(
            "Ollama model ready" if adaptive_ok else "Ollama model unavailable"
        )
        with st.expander("Technical availability details", expanded=False):
            st.markdown("**MiniLM**")
            st.write(dense_status)
            st.markdown("**Ollama**")
            st.write(adaptive_status)
    saved = _saved_payload(url, title, body)
    primary, secondary = st.columns([2, 1])
    localize_clicked = primary.button(
        "Localize issue", type="primary", width="stretch", key="run_live"
    )
    load_clicked = secondary.button(
        "Load saved result",
        width="stretch",
        disabled=saved is None,
        help="Load matching saved rankings without rerunning retrieval or inference.",
        key="load_live_saved",
    )
    if saved is not None:
        st.caption("A matching saved run is available. Loading it does not rerun localization.")
    if load_clicked and saved is not None:
        try:
            with st.status("Loading saved rankings and repository views…", expanded=True) as status:
                run = _load_saved_run(*saved)
                status.update(label="Saved run loaded", state="complete", expanded=False)
            st.session_state.live_run = run
        except Exception as error:
            st.error(f"Saved run could not be loaded: {type(error).__name__}: {error}")
    if localize_clicked:
        started = perf_counter()
        try:
            with st.status("Localizing issue…", expanded=True) as status:
                repository = parse_public_github_url(url)
                st.write("Validating repository and pinning an exact commit")
                if commit.strip():
                    resolved, branch = validate_commit(commit), "user-pinned"
                else:
                    repository, resolved, branch = resolve_default_branch_head(url)
                issue = live_input(repository, resolved, title, body)
                service = LiveLocalizationService(
                    LIVE_CACHE, embedding_cache=EmbeddingCache(LIVE_EMBEDDINGS)
                )
                st.write("Checking out source and building Python representations")
                prepared = service.prepare(issue)
                computed: dict[str, LiveMethodResult] = {}
                if "BM25" in methods:
                    st.write("Running BM25")
                    computed["bm25"] = service.bm25(prepared)
                encoder: SentenceTransformerEncoder | None = None
                if "Dense + Hybrid + Static Graph" in methods or "Adaptive Fixed" in methods:
                    encoder, dense_status = _local_encoder()
                if "Dense + Hybrid + Static Graph" in methods:
                    if encoder is None:
                        st.warning(f"Dense, Hybrid, and Static Graph were skipped. {dense_status}")
                    else:
                        st.write("Running Dense, Hybrid, and Static Graph")
                        dense, hybrid, graph = service.dense_hybrid_graph(prepared, encoder)
                        computed.update({"dense": dense, "hybrid": hybrid, "graph": graph})
                adaptive: AdaptiveLocalizationResult | None = None
                if "Adaptive Fixed" in methods:
                    if encoder is None:
                        st.warning("Adaptive was skipped because its Hybrid seed needs MiniLM.")
                    elif not adaptive_ok:
                        st.warning(f"Adaptive was skipped. {adaptive_status}")
                    else:
                        st.write("Running prompted Adaptive navigation")
                        adaptive = _run_adaptive(prepared, encoder, endpoint, model)
                artifact = save_live_artifact(
                    LIVE_RESULTS,
                    url,
                    branch,
                    prepared,
                    computed,
                    adaptive=adaptive.model_dump(mode="json") if adaptive is not None else None,
                )
                st.session_state.live_run = (
                    prepared,
                    computed,
                    adaptive,
                    artifact,
                    branch,
                    "New localization computed",
                    perf_counter() - started,
                )
                status.update(label="Localization complete", state="complete", expanded=False)
        except Exception as error:
            st.error(f"Localization could not be completed: {type(error).__name__}: {error}")
    stored_run = cast(tuple[Any, ...] | None, st.session_state.get("live_run"))
    if stored_run is not None:
        _render_results(stored_run)


def render_live_repository_sidebar() -> None:
    """Describe only the active live-repository workflow in the sidebar."""
    url = str(st.session_state.get("live_url", PRESET_URL))
    try:
        repository = parse_public_github_url(url)
    except ValueError:
        repository = url.strip() or "No repository selected"
    methods = cast(list[str], st.session_state.get("live_methods", []))
    st.sidebar.markdown("### LIVE REPOSITORY")
    st.sidebar.markdown("**Repository**")
    st.sidebar.write(repository)
    st.sidebar.markdown("**Scope:** Python source localization")
    if methods:
        st.sidebar.markdown("**Selected methods**")
        for method in methods:
            st.sidebar.write(f"• {method}")
    st.sidebar.markdown("### LIVE PIPELINE")
    st.sidebar.markdown(
        "1. Issue and Repository\n"
        "2. Source Snapshot\n"
        "3. Python Structure\n"
        "4. Repository Graph\n"
        "5. Search Documents\n"
        "6. Retrieval Results\n"
        "7. Static Graph, if enabled\n"
        "8. Adaptive Navigation, if enabled\n"
        "9. Final Candidates"
    )
    st.sidebar.caption("Detailed technical information is collapsed by default.")
