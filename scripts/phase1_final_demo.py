"""Official offline entry point for the GraphFix Phase-1 final demonstration.

The validated walkthrough remains implemented in ``phase1_pipeline_explorer.py``. This
entry point reuses it and changes only final-demonstration presentation text.
"""

from __future__ import annotations

import inspect
import json
import runpy
from pathlib import Path
from typing import Any, cast

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
WALKTHROUGH = ROOT / "scripts" / "phase1_pipeline_explorer.py"
FULL560_SUMMARY = (
    ROOT
    / "results"
    / "adaptive_navigation_experiments"
    / "fixed_pool_560_analysis"
    / "full560_summary.json"
)


def readable_node(node_id: str) -> str:
    """Return a concise display name while preserving full IDs in advanced views."""
    if node_id == "STOP":
        return node_id
    parts = node_id.split(":", 3)
    node_type = parts[0] if parts else ""
    if node_type in {"function", "method"} and len(parts) > 2:
        return f"{parts[2].rsplit('.', 1)[-1]}()"
    if node_type == "file" and len(parts) > 1:
        return parts[1]
    return node_id.rsplit(":", 1)[-1]


def run_validated_walkthrough(*, show_sidebar: bool = True) -> None:
    """Run the historical implementation with final-demo-only labels."""
    streamlit = cast(Any, st)
    original_page_config = streamlit.set_page_config
    original_title = streamlit.title
    original_caption = streamlit.caption
    original_header = streamlit.header
    original_write = streamlit.write
    original_warning = streamlit.warning
    original_markdown = streamlit.markdown
    original_json = streamlit.json
    original_code = streamlit.code
    original_expander = streamlit.expander
    original_dataframe = streamlit.dataframe
    original_graphviz_chart = streamlit.graphviz_chart
    original_selectbox = streamlit.selectbox
    original_sidebar_selectbox = streamlit.sidebar.selectbox
    original_columns = streamlit.columns
    original_container = streamlit.container
    first_caption = True
    current_stage = 0
    selected_parsed_file: str | None = None
    stage4_file_graph_rendered = False
    suppress_next_stage4_graph = False
    primary_trace_shown = False
    selected_instance = "vllm-project__vllm-9390"
    stage9_dataframes = 0
    stage7_dataframes = 0
    retrieval_heading: str | None = None

    takeaways = {
        1: "GraphFix begins only with the issue description and repository context; "
        "gold modification locations are withheld.",
        2: "Exact reconstruction prevents evaluation against newer code that did not "
        "exist for the original issue.",
        3: "AST parsing converts raw source into structured information that can be "
        "indexed and connected.",
        4: "The graph provides structural paths that retrieval alone cannot represent.",
        5: "SearchDocuments provide the common representation used by lexical and "
        "semantic retrieval.",
        6: "Hybrid combines complementary lexical and semantic evidence and is strongest "
        "in the completed full-560 deterministic evaluation.",
        7: "The full-560 evaluation shows that blind structural expansion can introduce "
        "ranking noise compared with Hybrid.",
        8: "Adaptive navigation explores graph structure selectively rather than expanding "
        "every neighbor.",
    }

    def final_page_config(*args: Any, **kwargs: Any) -> Any:
        # The two-tab outer application has already configured the page. The reused
        # historical walkthrough still calls set_page_config, which must be a no-op here.
        return None

    def final_title(_label: str, *args: Any, **kwargs: Any) -> Any:
        return original_title("GraphFix — Phase-1 Final Demonstration", *args, **kwargs)

    def final_caption(body: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal first_caption
        if first_caption:
            first_caption = False
            body = "Graph-Guided Software Issue Localization"
            result = original_caption(body, *args, **kwargs)
            original_write(
                "GraphFix localizes the files and functions likely to require modification "
                "for a software issue by combining lexical retrieval, semantic retrieval, "
                "repository structure, and selective graph-guided navigation."
            )
            original_markdown(
                "**Issue → Historical Code → AST → Repository Graph → Retrieval → "
                "Graph Navigation → Localization → Post-Hoc Evaluation**"
            )
            original_caption("Offline frozen demonstration — no live model or network required.")
            return result
        return original_caption(body, *args, **kwargs)

    def final_header(body: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal current_stage
        labels = {
            "1. Dataset Input": "1. Dataset / Issue Input",
            "3. Python AST Parsing": "3. AST Parsing",
            "6. Candidate Retrieval": "6. Candidate Retrieval — BM25, Dense, Hybrid",
        }
        label = labels.get(body, body)
        if isinstance(label, str) and label[:1].isdigit():
            next_stage = int(label.split(".", 1)[0])
            if current_stage in takeaways:
                original_caption(f"Takeaway: {takeaways[current_stage]}")
            current_stage = next_stage
        result = original_header(label, *args, **kwargs)
        if current_stage == 2:
            original_write(
                "The repository is reconstructed at the benchmark-provided base_commit: "
                "the pre-fix code snapshot used for localization. This does not assume "
                "that the commit exactly matches the issue-creation timestamp."
            )
        elif current_stage == 3:
            original_write(
                "Python source is parsed into structured program entities and relationships."
            )
        elif current_stage == 4:
            original_write(
                "The repository graph connects files and symbols through structural "
                "relationships such as CONTAINS, DEFINED_IN, IMPORTS, CALLS and INHERITS."
            )
            original_write(
                "The repository graph is built from the complete parsed repository. The "
                "view below shows the local neighborhood of the selected symbol from the "
                "file chosen in Stage 3."
            )
            with original_expander("Relationship legend"):
                original_markdown(
                    "- **CONTAINS** — hierarchy\n"
                    "- **DEFINED_IN** — symbol to file\n"
                    "- **IMPORTS** — module dependency\n"
                    "- **CALLS** — callable relationship\n"
                    "- **INHERITS** — class inheritance"
                )
        elif current_stage == 5:
            original_write(
                "The AST-derived ParsedRepository is used to create searchable file-level "
                "and callable-level SearchDocuments. RepositoryGraph and SearchDocuments "
                "are separate representations derived from the parsed repository."
            )
        elif current_stage == 6:
            original_write(
                "Hybrid combines complementary lexical and semantic rankings using "
                "Reciprocal Rank Fusion."
            )
        elif current_stage == 7:
            original_write(
                "Static Graph expands structural neighbors uniformly. It is included to "
                "test whether graph information alone improves localization."
            )
            streamlit.info(
                "Full-560 finding: blind structural expansion can introduce ranking noise. "
                "This is an aggregate conclusion, not a claim from this single example."
            )
        elif current_stage == 8:
            original_write(
                "Instead of expanding every graph neighbor, the Adaptive agent selects "
                "repository-navigation actions based on the current evidence."
            )
            streamlit.info(
                "Saved Phase-1 navigation trace — no live LLM inference is required "
                "during the demonstration."
            )
        elif current_stage == 10:
            original_write("Did the localization match the known modification locations?")
        return result

    def render_sidebar_case(value: Any) -> None:
        nonlocal selected_instance
        selected_instance = str(value)
        match = __import__("re").fullmatch(r"(.+)__(.+)-(\d+)", selected_instance)
        repository = f"{match.group(1)}/{match.group(2)}" if match is not None else "unknown"
        streamlit.sidebar.markdown("### DEMO CASE")
        streamlit.sidebar.markdown(f"**Repository:** `{repository}`")
        streamlit.sidebar.markdown(f"**Instance:** `{selected_instance}`")
        streamlit.sidebar.markdown("**Mode:** Offline / Frozen artifacts")
        streamlit.sidebar.markdown("**Primary configuration:** Adaptive — Fixed candidate pool")
        streamlit.sidebar.caption("Evaluation cohort: 560 / 560 issues")
        streamlit.sidebar.markdown("### PIPELINE")
        streamlit.sidebar.markdown(
            "1. Issue Input\n"
            "2. Historical Reconstruction\n"
            "3. AST Parsing\n"
            "4. Repository Graph\n"
            "5. Search Documents\n"
            "6. Candidate Retrieval\n"
            "7. Static Graph\n"
            "8. Adaptive Navigation\n"
            "9. Final Localization\n"
            "10. Post-Hoc Evaluation"
        )
        streamlit.sidebar.caption("Advanced details are collapsed by default.")

    def final_selectbox(label: str, *args: Any, **kwargs: Any) -> Any:
        nonlocal selected_parsed_file
        value = original_selectbox(label, *args, **kwargs)
        if label == "Parsed Python file":
            selected_parsed_file = str(value)
        if label in {
            "Parsed Python file",
            "Graph seed node",
            "Graph node",
            "Complete SearchDocument",
        }:
            full_value = getattr(value, "document_id", value)
            original_caption(f"Complete selection: `{full_value}`")
        return value

    def final_sidebar_selectbox(label: str, *args: Any, **kwargs: Any) -> Any:
        if not show_sidebar:
            options = args[0] if args else kwargs["options"]
            index = int(kwargs.get("index", 0))
            return options[index]
        value = original_sidebar_selectbox(label, *args, **kwargs)
        if label == "Demo instance":
            render_sidebar_case(value)
        return value

    def final_json(body: Any, *args: Any, **kwargs: Any) -> Any:
        if current_stage == 1 and isinstance(body, dict) and "instance_id" in body:
            original_markdown(f"**Repository:** `{body.get('repo', '')}`")
            original_markdown(f"**Instance ID:** `{body.get('instance_id', '')}`")
            commit = str(body.get("base_commit", ""))
            original_markdown(f"**Historical commit:** `{commit[:10]}`")
            original_markdown("**Issue description**")
            with original_container(border=True):
                issue = str(body.get("problem_statement", ""))
                preview = issue if len(issue) <= 1_200 else issue[:1_200].rstrip() + "…"
                original_write(preview)
            with original_expander("Advanced — Raw gold-free benchmark input"):
                return original_json(body, *args, **kwargs)
        if current_stage == 10 and isinstance(body, dict) and body.get("completed") == 20:
            full560 = cast(
                dict[str, Any],
                json.loads(FULL560_SUMMARY.read_text(encoding="utf-8")),
            )
            provenance = full560["provenance"]
            integrity = full560["integrity"]
            efficiency = full560["efficiency"]
            context = {
                "cohort_size": provenance["cohort_size"],
                "adaptive_artifacts": integrity["combined_unique_count"],
                "candidate_pool": provenance["configuration"]["candidate_pool"],
                "file_function_episodes": efficiency["episodes"],
                "missing": len(integrity["missing_ids"]),
                "duplicates": len(integrity["duplicate_ids"]),
                "unexpected": len(integrity["unexpected_ids"]),
            }
            return original_json(context, *args, **kwargs)
        return original_json(body, *args, **kwargs)

    def final_code(body: Any, *args: Any, **kwargs: Any) -> Any:
        if current_stage == 3 and kwargs.get("language") == "python":
            with original_expander("Advanced — Parsed source example"):
                return original_code(body, *args, **kwargs)
        return original_code(body, *args, **kwargs)

    def final_expander(label: str, *args: Any, **kwargs: Any) -> Any:
        labels = {
            (1, "Advanced Details"): "Advanced — Input boundary",
            (2, "Advanced Details"): "Advanced — Reconstruction details",
            (3, "Advanced Details"): "Advanced — Parsed entities",
            (4, "Advanced Graph Details"): "Advanced — Graph controls and node inspection",
            (5, "Advanced Details"): "Advanced — Full SearchDocument contents",
            (6, "Advanced Details"): "Advanced — Full retrieval rankings",
            (7, "Advanced Details"): "Advanced — Static expansion details",
            (8, "Advanced Trace Details"): "Advanced — Full technical Adaptive trace",
            (9, "Advanced Details"): "Advanced — Full localization rankings",
            (10, "Advanced Evaluation Details"): "Advanced — Full gold comparison",
            (10, "Advanced Details: full evaluation context"): (
                "Advanced — Full-560 evaluation context"
            ),
        }
        kwargs.setdefault("expanded", False)
        return original_expander(labels.get((current_stage, label), label), *args, **kwargs)

    def final_dataframe(data: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal primary_trace_shown, retrieval_heading, stage7_dataframes, stage9_dataframes
        if current_stage == 6 and retrieval_heading is not None and isinstance(data, list):
            concise = [
                {
                    "Rank": row.get("Rank"),
                    "Candidate": row.get("Callable") or row.get("File"),
                    "Score": row.get("Score"),
                }
                for row in data
            ]
            descriptions = {
                "BM25": "Lexical matching",
                "Dense": "Semantic similarity",
                "Hybrid": "Reciprocal Rank Fusion of lexical + semantic evidence",
            }
            method = retrieval_heading
            retrieval_heading = None
            with original_container(border=True):
                original_markdown(f"#### {method}")
                original_caption(descriptions[method])
                return original_dataframe(concise, *args, **kwargs)
        if current_stage == 8 and not primary_trace_shown and isinstance(data, list):
            primary_trace_shown = True
            concise = [
                {
                    "Step": row.get("Step"),
                    "Action": row.get("Action"),
                    "Target": readable_node(str(row.get("Target node") or "STOP")),
                    "New evidence?": row.get("New evidence"),
                    "Result / observation": row.get("Observation / relation"),
                }
                for row in data
            ]
            streamlit.info(
                "This representative trace exposes the Phase-1 efficiency limitation: "
                "the first inspection contributes new evidence, while later repeated "
                "actions may add no new evidence."
            )
            return original_dataframe(concise, *args, **kwargs)
        if current_stage == 7:
            stage7_dataframes += 1
            if stage7_dataframes == 1:
                return None
        if current_stage == 9:
            stage9_dataframes += 1
            if stage9_dataframes <= 2:
                return None
        return original_dataframe(data, *args, **kwargs)

    def final_markdown(body: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal retrieval_heading
        retrieval_labels = {
            "**BM25 — file Top 5**": "BM25",
            "**Dense — file Top 5**": "Dense",
            "**Hybrid — file Top 5**": "Hybrid",
        }
        if current_stage == 6 and body in retrieval_labels:
            retrieval_heading = retrieval_labels[body]
            return None
        if current_stage == 9 and body in {
            "**Files: Hybrid ranking → Adaptive ranking**",
            "**Functions/methods: Hybrid ranking → Adaptive ranking**",
        }:
            return None
        return original_markdown(body, *args, **kwargs)

    def render_file_graph(caller_globals: dict[str, Any]) -> None:
        """Render one neighborhood from the existing graph for the Stage-3 file."""
        nonlocal suppress_next_stage4_graph
        graph = caller_globals["graph"]
        selected_file = selected_parsed_file
        if selected_file is None:
            return
        type_order = {"function": 0, "method": 1, "class": 2, "file": 3}
        candidates = [
            node
            for node in graph.nodes
            if node.file_path == selected_file and node.node_type in type_order
        ]
        hybrid_rankings = caller_globals["hybrid_rankings"]["function"]
        hybrid_order = {
            candidate["document_id"]: index
            for index, candidate in enumerate(hybrid_rankings)
            if candidate["file_path"] == selected_file
        }
        candidates.sort(
            key=lambda node: (
                not (selected_instance == "vllm-project__vllm-9390" and node.name == "benchmark"),
                hybrid_order.get(node.node_id, len(hybrid_order) + 1),
                type_order[node.node_type],
                node.qualified_name or node.name,
                node.node_id,
            )
        )
        if not candidates:
            original_warning(f"No graph node is available for `{selected_file}`.")
            return

        def node_label(node_id: str) -> str:
            node = graph.get_node(node_id)
            if node is None:
                return node_id
            suffix = "()" if node.node_type in {"function", "method"} else ""
            return f"{node.name}{suffix} — {node.node_type}"

        node_ids = [node.node_id for node in candidates]
        selected_node_id = original_selectbox(
            "Graph symbol",
            node_ids,
            index=0,
            format_func=node_label,
            key=f"final_graph_symbol::{selected_file}",
        )
        selected_node = graph.get_node(selected_node_id)
        assert selected_node is not None
        original_markdown(f"**Selected file:** `{selected_file}`")
        original_markdown(
            f"**Selected graph symbol:** `{node_label(selected_node_id).rsplit(' — ', 1)[0]}`"
        )
        neighborhood_edges = caller_globals["neighborhood_edges"]
        graphviz_source = caller_globals["graphviz_source"]
        edges, limited = neighborhood_edges(
            graph,
            selected_node_id,
            1,
            set(caller_globals["EDGE_TYPES"]),
            set(caller_globals["NODE_TYPES"]),
        )
        original_graphviz_chart(graphviz_source(graph, edges, selected_node_id), width="stretch")
        if not edges:
            original_caption("This symbol has no matching neighbors under the current local view.")
        elif limited:
            original_caption("Display capped at the existing graph-edge limit.")
        suppress_next_stage4_graph = True

    def final_graphviz_chart(*args: Any, **kwargs: Any) -> Any:
        nonlocal suppress_next_stage4_graph
        if current_stage == 4 and suppress_next_stage4_graph:
            suppress_next_stage4_graph = False
            return None
        return original_graphviz_chart(*args, **kwargs)

    class MetricColumnProxy:
        """Keep numeric graph metrics while moving long identifiers to text."""

        def __init__(self, column: Any, *, hidden: bool = False) -> None:
            self.column = column
            self.hidden = hidden

        def metric(self, label: str, value: Any, *args: Any, **kwargs: Any) -> Any:
            if self.hidden:
                return None
            if label == "Selected root":
                node_id = str(value)
                parts = node_id.split(":", 3)
                node_type = parts[0] if parts else "unknown"
                file_path = parts[1] if len(parts) > 1 else "not available"
                qualified = parts[2] if len(parts) > 2 else node_id
                symbol = qualified.rsplit(".", 1)[-1]
                original_markdown(f"**Selected symbol:** `{symbol}()`")
                original_markdown(f"**Node type:** `{node_type}`")
                original_markdown(f"**File:** `{file_path}`")
                with original_expander("Advanced — Graph node details"):
                    original_markdown("**Full graph node ID**")
                    original_code(node_id)
                return None
            if label == "Root node type":
                return None
            return self.column.metric(label, value, *args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.column, name)

    def final_columns(spec: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal stage4_file_graph_rendered
        columns = original_columns(spec, *args, **kwargs)
        if current_stage == 4 and spec == 2 and not stage4_file_graph_rendered:
            caller = inspect.currentframe()
            assert caller is not None and caller.f_back is not None
            render_file_graph(caller.f_back.f_globals)
            stage4_file_graph_rendered = True
        if current_stage == 4 and spec == 4:
            return [MetricColumnProxy(column, hidden=True) for column in columns]
        return columns

    def final_write(*args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], str) and "Adaptive n=20 pilot is descriptive" in args[0]:
            args = (
                "Hybrid is the strongest deterministic retrieval configuration. On the "
                "frozen full-560 cohort, Adaptive Fixed improves ranking within Hybrid's "
                "Top-10 pool; this does not prove global superiority.",
                *args[1:],
            )
        return original_write(*args, **kwargs)

    def final_warning(body: Any, *args: Any, **kwargs: Any) -> Any:
        if body == "Saved pilot trace only — this walkthrough does not call Ollama.":
            return None
        if isinstance(body, str) and body.startswith(
            "Bounded expansion now permits observed outside-pool candidates to enter"
        ):
            body = (
                "Bounded candidate expansion produced no first-relevant-rank improvements "
                "in the evaluated n=20/n=50 ablations and introduced several degradations. "
                "Fixed pool therefore remains the primary Phase-1 configuration."
            )
        return original_warning(body, *args, **kwargs)

    streamlit.set_page_config = final_page_config
    streamlit.title = final_title
    streamlit.caption = final_caption
    streamlit.header = final_header
    streamlit.write = final_write
    streamlit.warning = final_warning
    streamlit.markdown = final_markdown
    streamlit.json = final_json
    streamlit.code = final_code
    streamlit.expander = final_expander
    streamlit.dataframe = final_dataframe
    streamlit.graphviz_chart = final_graphviz_chart
    streamlit.selectbox = final_selectbox
    streamlit.sidebar.selectbox = final_sidebar_selectbox
    streamlit.columns = final_columns
    try:
        runpy.run_path(str(WALKTHROUGH), run_name="__main__")
    finally:
        streamlit.set_page_config = original_page_config
        streamlit.title = original_title
        streamlit.caption = original_caption
        streamlit.header = original_header
        streamlit.write = original_write
        streamlit.warning = original_warning
        streamlit.markdown = original_markdown
        streamlit.json = original_json
        streamlit.code = original_code
        streamlit.expander = original_expander
        streamlit.dataframe = original_dataframe
        streamlit.graphviz_chart = original_graphviz_chart
        streamlit.selectbox = original_selectbox
        streamlit.sidebar.selectbox = original_sidebar_selectbox
        streamlit.columns = original_columns


def show_phase1_summary() -> None:
    """Show a compact post-boundary summary loaded from frozen full-560 results."""
    payload = cast(
        dict[str, Any],
        json.loads(FULL560_SUMMARY.read_text(encoding="utf-8")),
    )
    adaptive = payload["adaptive_fixed_metrics"]
    deltas = payload["hybrid_to_adaptive_fixed_deltas"]
    resources = payload["efficiency"]
    hybrid = {key: adaptive[key] - deltas[key] for key in deltas}

    st.divider()
    st.subheader("Phase-1 Evaluation Summary")
    st.caption("FULL 560-ISSUE COHORT RESULT")
    st.success(
        "On the frozen full-560 cohort, Adaptive Fixed achieves the strongest ranking "
        "performance among the evaluated configurations."
    )
    st.markdown("**Definitive full-560 fixed-pool Adaptive evaluation**")
    first_metric_row = st.columns(2)
    second_metric_row = st.columns(2)
    for column, label, baseline, final in (
        (
            first_metric_row[0],
            "FILE Hit@1",
            hybrid["file_hit_at_1"],
            adaptive["file_hit_at_1"],
        ),
        (first_metric_row[1], "FILE MRR", hybrid["file_mrr"], adaptive["file_mrr"]),
        (
            second_metric_row[0],
            "FUNCTION Hit@1",
            hybrid["function_hit_at_1"],
            adaptive["function_hit_at_1"],
        ),
        (
            second_metric_row[1],
            "FUNCTION MRR",
            hybrid["function_mrr"],
            adaptive["function_mrr"],
        ),
    ):
        column.metric(label, f"{final:.3f}")
        column.caption(
            f"Adaptive Fixed\n\nHybrid baseline: {baseline:.3f}\n\n{baseline:.3f} → {final:.3f}"
        )
    efficiency = st.columns(2)
    efficiency[0].metric("Repetition rate", f"{resources['repeated_action_target_rate']:.2%}")
    efficiency[1].metric("New-evidence rate", f"{resources['new_evidence_rate']:.2%}")
    st.subheader("What Phase 1 established")
    st.markdown(
        "1. Hybrid is the strongest deterministic retrieval configuration.\n"
        "2. Adaptive Fixed improves Hit@1, Recall@5, and MRR at both levels.\n"
        "3. Recall@10 is unchanged: ordering improves within the fixed pool, not coverage.\n"
        "4. Prompted navigation remains inefficient: 70.36% repeated action-targets and "
        "a 19.19% new-evidence rate.\n"
        "5. Expansion produced no first-rank improvements in n=20/n=50 ablations and "
        "introduced degradations, so fixed pool remains primary."
    )
    st.info(
        "Phase-2 motivation: learn when additional graph exploration is actually necessary "
        "and when the system should stop. Phase-2 methods are future work and are not "
        "implemented in this demonstration."
    )
    st.caption(
        "The full-560 result is cohort- and configuration-specific; it does not prove "
        "global superiority. All values above are read from frozen artifacts."
    )


st.set_page_config(page_title="GraphFix — Phase-1 Final Demonstration", layout="wide")
mode_labels = ["Benchmark Evidence", "Live Repository"]
active_mode = cast(str, st.session_state.get("graphfix_demo_mode", mode_labels[0]))
benchmark_tab, live_tab = st.tabs(
    mode_labels,
    key="graphfix_demo_mode",
    default=active_mode,
    on_change="rerun",
)
with benchmark_tab:
    run_validated_walkthrough(show_sidebar=active_mode == "Benchmark Evidence")
    show_phase1_summary()
with live_tab:
    from live_repository_tab import render_live_repository_sidebar, render_live_repository_tab

    render_live_repository_tab()
if active_mode == "Live Repository":
    render_live_repository_sidebar()
