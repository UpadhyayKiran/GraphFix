"""Analyze the frozen adaptive pilot from saved artifacts without running localization."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, cast

from datasets import load_from_disk  # type: ignore[import-untyped]

from graphfix.datasets import normalize_locbench_records
from graphfix.evaluation import (
    aggregate_metrics,
    evaluate_rankings,
    extract_gold_locations,
    normalize_repository_path,
)
from graphfix.pipeline import AdaptiveLocalizationResult

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "results" / "localization_benchmark" / "evaluation_cohort.json"
SNAPSHOT = ROOT / "data" / "raw" / "locbench_v1" / "dataset"
ADAPTIVE_RESULTS = ROOT / "results" / "adaptive_navigation"
FULL_COMPARISON = ROOT / "results" / "retrieval_comparison" / "full_evaluation"
PILOT_OUTPUT = ROOT / "results" / "review2" / "pilot_summary"
PHASE1_SUMMARY_OUTPUT = ROOT / "results" / "phase1_evaluation_summary"
PILOT_SIZE = 20


def resolve_pilot_cohort(path: Path = COHORT, limit: int = PILOT_SIZE) -> tuple[str, ...]:
    """Resolve the deterministic pilot as the first IDs in the frozen cohort."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = tuple(cast(list[str], payload["included_ids"])[:limit])
    if len(ids) != limit or len(set(ids)) != limit:
        raise ValueError(f"Expected {limit} unique frozen cohort IDs")
    return ids


def navigation_stats(navigation: dict[str, Any]) -> dict[str, Any]:
    """Derive novelty and repetition from accepted trace steps only."""
    evidence: set[tuple[str, str, str, str]] = set()
    action_targets: set[tuple[str, str]] = set()
    unique_nodes: set[str] = set()
    new_yes = 0
    repeated = 0
    trace_rows: list[dict[str, Any]] = []
    for index, step in enumerate(navigation["trace"], start=1):
        pair = (step["action"], step["node_id"])
        is_repeated = pair in action_targets
        repeated += int(is_repeated)
        action_targets.add(pair)
        additions = 0
        relationships: list[str] = []
        for observed in step["observed_nodes"]:
            item = (
                observed["node_id"],
                observed["relationship"],
                observed["direction"],
                observed["snippet"],
            )
            if item not in evidence:
                evidence.add(item)
                additions += 1
            unique_nodes.add(observed["node_id"])
            relationships.append(
                f"{observed['direction']} {observed['relationship']} -> {observed['node_id']}"
            )
        is_new = additions > 0
        new_yes += int(is_new)
        trace_rows.append(
            {
                "step": index,
                "action": step["action"],
                "node_id": step["node_id"],
                "observation": "; ".join(relationships) or step["message"],
                "new_evidence": is_new,
                "repeated_action_target": is_repeated,
                "cumulative_unique_nodes": len(unique_nodes),
                "validation_errors": step["validation_errors"],
                "invalid_attempts": step["invalid_attempts"],
                "model_calls": step["model_calls"],
            }
        )
    steps = int(navigation["actual_steps"])
    return {
        "steps": steps,
        "model_calls": int(navigation["model_calls"]),
        "retries": int(navigation["retry_count"]),
        "validation_failures": int(navigation["validation_failures"]),
        "unique_nodes": len(unique_nodes),
        "repeated_action_targets": repeated,
        "new_evidence_yes": new_yes,
        "new_evidence_no": steps - new_yes,
        "new_evidence_rate": new_yes / steps if steps else 0.0,
        "repetition_rate": repeated / steps if steps else 0.0,
        "stopping_reason": navigation["stopping_reason"],
        "explicit_stop": bool(navigation["stopped_by_controller"]),
        "trace": trace_rows,
    }


def _rank_change(before: int | None, after: int | None) -> str:
    """Classify whether adaptive reranking improved, preserved, or degraded a rank."""
    if before is None and after is None:
        return "UNCHANGED"
    if before is None or (after is not None and after < before):
        return "IMPROVED"
    if after is None or after > before:
        return "DEGRADED"
    return "UNCHANGED"


def _interpretation(before: int | None, outcome: str) -> str:
    """Give a compact outcome interpretation using the initial rank and rank change."""
    if outcome == "IMPROVED":
        return "RESCUE"
    if outcome == "DEGRADED":
        return "CORRUPTION"
    return "UNNECESSARY_AGENT" if before is not None and before <= 5 else "NO_EFFECT"


def _outside_pool_diagnostics(
    payload: dict[str, Any], gold_files: set[str], gold_functions: set[str]
) -> tuple[int, list[dict[str, Any]]]:
    """Find observed graph nodes outside the fixed Hybrid pool and test gold relevance."""
    initial_ids = {
        candidate["document_id"]
        for level in ("file_navigation", "function_navigation")
        for candidate in payload[level]["initial_ranking"]
    }
    outside: dict[str, dict[str, Any]] = {}
    for level in ("file_navigation", "function_navigation"):
        for step in payload[level]["trace"]:
            for node in step["observed_nodes"]:
                if node["node_id"] not in initial_ids:
                    outside[node["node_id"]] = node
    relevant: list[dict[str, Any]] = []
    for node in outside.values():
        path = normalize_repository_path(node["file_path"])
        levels: list[str] = []
        if path in gold_files:
            levels.append("file")
        function = f"{path}:{node['name'].casefold()}"
        if node["node_type"] in ("function", "method") and function in gold_functions:
            levels.append("function")
        if levels:
            relevant.append({"node_id": node["node_id"], "gold_levels": levels})
    return len(outside), relevant


def analyze_pilot(
    cohort_path: Path = COHORT,
    snapshot: Path = SNAPSHOT,
    adaptive_dir: Path = ADAPTIVE_RESULTS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Analyze persisted gold-free results, then apply gold post hoc."""
    instance_ids = resolve_pilot_cohort(cohort_path)
    raw = load_from_disk(str(snapshot))
    dataset = normalize_locbench_records(
        cast(Any, raw), raw_fields=cast(list[str], raw.column_names)
    )
    instances = {item.input.instance_id: item for item in dataset.instances}
    rows: list[dict[str, Any]] = []
    hybrid_metrics = []
    adaptive_metrics = []
    for instance_id in instance_ids:
        result_path = adaptive_dir / f"{instance_id}.adaptive.localization.json"
        result_text = result_path.read_text(encoding="utf-8")
        raw_payload = json.loads(result_text)
        result = AdaptiveLocalizationResult.model_validate_json(result_text)
        instance = instances[instance_id]
        gold = extract_gold_locations(instance.gold)
        hybrid = evaluate_rankings(
            result.file_navigation.initial_ranking,
            result.function_navigation.initial_ranking,
            gold,
        )
        adaptive = evaluate_rankings(
            result.file_navigation.final_ranking,
            result.function_navigation.final_ranking,
            gold,
        )
        hybrid_metrics.append(hybrid)
        adaptive_metrics.append(adaptive)
        file_stats = navigation_stats(raw_payload["file_navigation"])
        function_stats = navigation_stats(raw_payload["function_navigation"])
        file_change = _rank_change(
            hybrid.file.first_relevant_rank, adaptive.file.first_relevant_rank
        )
        function_change = _rank_change(
            hybrid.function.first_relevant_rank, adaptive.function.first_relevant_rank
        )
        limited = (
            hybrid.file.first_relevant_rank is None
            or hybrid.function.first_relevant_rank is None
        )
        combined_new = file_stats["new_evidence_yes"] + function_stats["new_evidence_yes"]
        behavior = "CANDIDATE_POOL_LIMITED" if limited else "PARTIALLY_PRODUCTIVE"
        if not limited and combined_new == 0:
            behavior = "UNPRODUCTIVE"
        outside_count, relevant_outside = _outside_pool_diagnostics(
            raw_payload, set(gold.files), set(gold.functions)
        )
        rows.append(
            {
                "instance_id": instance_id,
                "repository": result.repository,
                "issue": result.issue,
                "requested_commit": result.base_commit,
                "verified_commit": result.checked_out_commit,
                "checkout_path": result.checkout_path,
                "parser_stats": result.parser_stats.model_dump(),
                "graph_stats": result.graph_stats.model_dump(),
                "hybrid_file_rank": hybrid.file.first_relevant_rank,
                "adaptive_file_rank": adaptive.file.first_relevant_rank,
                "hybrid_function_rank": hybrid.function.first_relevant_rank,
                "adaptive_function_rank": adaptive.function.first_relevant_rank,
                "file_rank_change": file_change,
                "function_rank_change": function_change,
                "file_interpretation": _interpretation(
                    hybrid.file.first_relevant_rank, file_change
                ),
                "function_interpretation": _interpretation(
                    hybrid.function.first_relevant_rank, function_change
                ),
                "hybrid_file_metrics": hybrid.file.model_dump(),
                "adaptive_file_metrics": adaptive.file.model_dump(),
                "hybrid_function_metrics": hybrid.function.model_dump(),
                "adaptive_function_metrics": adaptive.function.model_dump(),
                "file_navigation": file_stats,
                "function_navigation": function_stats,
                "candidate_pool_limited": limited,
                "outside_pool_nodes_observed": outside_count,
                "gold_relevant_outside_pool_nodes": relevant_outside,
                "behavior_label": behavior,
                "runtime_seconds": result.runtime_seconds,
                "rankings": {
                    "bm25": _load_rankings(instance_id, "bm25"),
                    "dense": _load_rankings(instance_id, "dense"),
                    "hybrid": _load_rankings(instance_id, "hybrid"),
                    "graph": _load_rankings(instance_id, "graph"),
                    "adaptive": {
                        "file": [x.model_dump() for x in result.file_navigation.final_ranking],
                        "function": [
                            x.model_dump() for x in result.function_navigation.final_ranking
                        ],
                    },
                },
            }
        )
    summary = _aggregate(rows, hybrid_metrics, adaptive_metrics)
    return rows, summary


def _load_rankings(instance_id: str, method: str) -> dict[str, Any] | None:
    """Load saved file/function rankings for one comparison method when available."""
    path = FULL_COMPARISON / method / f"{instance_id}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"file": payload["file_rankings"], "function": payload["function_rankings"]}


def _aggregate(
    rows: list[dict[str, Any]], hybrid: list[Any], adaptive: list[Any]
) -> dict[str, Any]:
    """Aggregate pilot metrics, navigation behavior, cost, and candidate-pool evidence."""
    levels = [row[level] for row in rows for level in ("file_navigation", "function_navigation")]
    file_levels = [row["file_navigation"] for row in rows]
    function_levels = [row["function_navigation"] for row in rows]

    def totals(items: list[dict[str, Any]]) -> dict[str, Any]:
        """Sum navigation counts and rates over a group of file/function episodes."""
        steps = sum(x["steps"] for x in items)
        repeated = sum(x["repeated_action_targets"] for x in items)
        yes = sum(x["new_evidence_yes"] for x in items)
        reasons = Counter(x["stopping_reason"] for x in items)
        return {
            "episodes": len(items),
            "steps": steps,
            "model_calls": sum(x["model_calls"] for x in items),
            "retries": sum(x["retries"] for x in items),
            "validation_failures": sum(x["validation_failures"] for x in items),
            "unique_node_observations": sum(x["unique_nodes"] for x in items),
            "repeated_action_targets": repeated,
            "new_evidence_yes": yes,
            "new_evidence_no": steps - yes,
            "new_evidence_rate": yes / steps,
            "no_new_evidence_rate": (steps - yes) / steps,
            "repetition_rate": repeated / steps,
            "stopping_reasons": dict(sorted(reasons.items())),
        }

    runtime = [float(row["runtime_seconds"]) for row in rows]
    h = aggregate_metrics(hybrid).model_dump()
    a = aggregate_metrics(adaptive).model_dump()
    outcomes = Counter(
        row[key] for row in rows for key in ("file_rank_change", "function_rank_change")
    )
    interpretations = Counter(
        row[key]
        for row in rows
        for key in ("file_interpretation", "function_interpretation")
    )
    return {
        "schema_version": 1,
        "pilot_size": len(rows),
        "completed": len(rows),
        "failed": 0,
        "experiment_level_reruns": 0,
        "configuration": {
            "model": "qwen2.5-coder:7b",
            "endpoint": "http://127.0.0.1:11434",
            "max_steps": 10,
            "max_retries": 2,
            "controller_timeout_seconds": 120,
        },
        "navigation": {
            "file": totals(file_levels),
            "function": totals(function_levels),
            "combined": totals(levels),
        },
        "step_distribution": {
            "mean": statistics.mean(x["steps"] for x in levels),
            "median": statistics.median(x["steps"] for x in levels),
            "minimum": min(x["steps"] for x in levels),
            "maximum": max(x["steps"] for x in levels),
        },
        "runtime_seconds": {
            "total": sum(runtime),
            "mean": statistics.mean(runtime),
            "median": statistics.median(runtime),
        },
        "metrics": {"hybrid": h, "adaptive": a},
        "rank_outcomes": dict(sorted(outcomes.items())),
        "interpretations": dict(sorted(interpretations.items())),
        "behavior_labels": dict(sorted(Counter(x["behavior_label"] for x in rows).items())),
        "candidate_pool": {
            "gold_file_outside_hybrid_top10": sum(
                x["hybrid_file_rank"] is None for x in rows
            ),
            "gold_function_outside_hybrid_top10": sum(
                x["hybrid_function_rank"] is None for x in rows
            ),
            "cases_observing_outside_pool_nodes": sum(
                x["outside_pool_nodes_observed"] > 0 for x in rows
            ),
            "outside_pool_nodes_observed": sum(
                x["outside_pool_nodes_observed"] for x in rows
            ),
            "cases_with_gold_relevant_outside_pool_node": sum(
                bool(x["gold_relevant_outside_pool_nodes"]) for x in rows
            ),
            "gold_relevant_outside_pool_nodes": sum(
                len(x["gold_relevant_outside_pool_nodes"]) for x in rows
            ),
        },
    }


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested per-instance analysis fields for CSV output."""
    flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
    for prefix in (
        "hybrid_file_metrics",
        "adaptive_file_metrics",
        "hybrid_function_metrics",
        "adaptive_function_metrics",
    ):
        for key, value in row[prefix].items():
            flat[f"{prefix}_{key}"] = value
    for prefix in ("file_navigation", "function_navigation"):
        for key, value in row[prefix].items():
            if key != "trace":
                flat[f"{prefix}_{key}"] = value
    return flat


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries with a shared schema to a UTF-8 CSV file."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(
    summary: dict[str, Any], pilot_output: Path, phase1_summary_output: Path
) -> None:
    """Regenerate deterministic SVG figures from saved comparison and pilot summaries."""
    pilot_output.mkdir(parents=True, exist_ok=True)
    phase1_summary_output.mkdir(parents=True, exist_ok=True)
    full_rows = list(csv.DictReader((FULL_COMPARISON / "comparison.csv").open(encoding="utf-8")))
    metrics = ("hit_at_1", "recall_at_5", "recall_at_10", "mrr")
    methods = ("bm25", "dense", "hybrid", "graph")
    labels = ("BM25", "Dense", "Hybrid", "Static Graph")
    found = {(r["method"], r["level"]): r for r in full_rows}
    groups = [
        f"{level.title()} {metric.replace('_', ' ').title()}"
        for level in ("file", "function")
        for metric in metrics
    ]
    full_series = {
        label: [
            float(found[(method, level)][metric])
            for level in ("file", "function")
            for metric in metrics
        ]
        for method, label in zip(methods, labels, strict=True)
    }
    _grouped_bar_svg(
        phase1_summary_output / "full560_method_comparison.svg",
        "GraphFix full 560-instance localization comparison",
        groups,
        full_series,
        maximum=0.7,
    )
    pilot_series = {
        method.title(): [
            summary["metrics"][method][f"{level}_{metric}"]
            for level in ("file", "function")
            for metric in metrics
        ]
        for method in ("hybrid", "adaptive")
    }
    _grouped_bar_svg(
        pilot_output / "pilot_hybrid_vs_adaptive.svg",
        "Hybrid vs Adaptive — n = 20 pilot",
        groups,
        pilot_series,
        maximum=0.7,
    )
    combined = summary["navigation"]["combined"]
    stops = combined["stopping_reasons"]
    _grouped_bar_svg(
        pilot_output / "pilot_trace_efficiency.svg",
        "Adaptive trace quality and stopping — n = 20 pilot",
        [
            "New evidence",
            "No new evidence",
            "Repeated pair",
            "Explicit STOP",
            "Invalid stop",
            "Max steps",
        ],
        {"Count": [
            combined["new_evidence_yes"],
            combined["new_evidence_no"],
            combined["repeated_action_targets"],
            stops.get("explicit_stop", 0),
            stops.get("invalid_response_stop", 0),
            stops.get("max_steps", 0),
        ]},
        maximum=260,
    )
    outcomes = summary["rank_outcomes"]
    _grouped_bar_svg(
        pilot_output / "pilot_rank_outcomes.svg",
        "Adaptive rank outcomes — n = 20 pilot (40 outcomes)",
        ["Improved", "Unchanged", "Degraded"],
        {
            "Outcomes": [
                outcomes.get("IMPROVED", 0),
                outcomes.get("UNCHANGED", 0),
                outcomes.get("DEGRADED", 0),
            ]
        },
        maximum=40,
    )


def _grouped_bar_svg(
    path: Path,
    title: str,
    groups: list[str],
    series: dict[str, list[float]],
    *,
    maximum: float,
) -> None:
    """Write a compact dependency-free grouped bar chart as deterministic SVG."""
    width, height = 1280, 620
    left, top, chart_width, chart_height = 85, 90, 1135, 400
    colors = ("#264653", "#2a9d8f", "#e9c46a", "#e76f51")
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="38" text-anchor="middle" '
            'font-family="sans-serif" font-size="24" font-weight="bold">'
            f"{title}</text>"
        ),
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = top + chart_height - chart_height * tick / 5
        parts.append(
            f'<line x1="{left}" y1="{y}" x2="{left + chart_width}" '
            f'y2="{y}" stroke="#d9d9d9"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 5}" text-anchor="end" '
            f'font-family="sans-serif" font-size="13">{value:.2f}</text>'
        )
    group_width = chart_width / len(groups)
    bar_width = min(34.0, group_width * 0.75 / len(series))
    for series_index, (name, values) in enumerate(series.items()):
        for group_index, value in enumerate(values):
            x = left + group_index * group_width + group_width * 0.125 + series_index * bar_width
            bar_height = chart_height * float(value) / maximum
            y = top + chart_height - bar_height
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 2:.2f}" '
                f'height="{bar_height:.2f}" fill="{colors[series_index]}"/>'
            )
        legend_x = left + series_index * 190
        parts.append(
            f'<rect x="{legend_x}" y="550" width="16" height="16" '
            f'fill="{colors[series_index]}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 23}" y="564" font-family="sans-serif" '
            f'font-size="15">{name}</text>'
        )
    for group_index, label in enumerate(groups):
        x = left + (group_index + 0.5) * group_width
        parts.append(
            f'<text x="{x:.2f}" y="510" text-anchor="middle" '
            'font-family="sans-serif" font-size="12" '
            f'transform="rotate(-18 {x:.2f} 510)">{label}</text>'
        )
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def write_outputs(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    pilot_output: Path,
    phase1_summary_output: Path,
) -> None:
    """Write historical pilot outputs and the Phase-1 comparison summary."""
    pilot_output.mkdir(parents=True, exist_ok=True)
    phase1_summary_output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        pilot_output / "adaptive_pilot_per_instance.csv",
        [_flatten_row(row) for row in rows],
    )
    (pilot_output / "adaptive_pilot_details.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    (pilot_output / "adaptive_pilot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    metric_rows = []
    for method in ("hybrid", "adaptive"):
        values = summary["metrics"][method]
        for level in ("file", "function"):
            metric_rows.append(
                {"method": method, "level": level, **{
                    key: values[f"{level}_{key}"]
                    for key in ("hit_at_1", "recall_at_5", "recall_at_10", "mrr")
                }}
            )
    _write_csv(pilot_output / "adaptive_pilot_metrics.csv", metric_rows)
    full_rows = list(csv.DictReader((FULL_COMPARISON / "comparison.csv").open(encoding="utf-8")))
    _write_csv(phase1_summary_output / "full560_method_comparison.csv", full_rows)
    findings = {
        "pilot_scope": "Descriptive fixed n=20 pilot; no population-level superiority claim.",
        "full_evaluation": "Hybrid is strongest on all reported full-560 metrics.",
        "adaptive": (
            "Prompted navigation occasionally improves ranks but is highly repetitive "
            "and weak at stopping."
        ),
        "candidate_pool": (
            "Adaptive reranks the original Hybrid Top 10 and cannot promote arbitrary "
            "graph-discovered nodes."
        ),
        "demo_cases": [
            "vllm-project__vllm-9390",
            "scikit-learn__scikit-learn-24145",
            "dmlc__dgl-5240",
        ],
    }
    (pilot_output / "review2_key_findings.json").write_text(
        json.dumps(findings, indent=2) + "\n", encoding="utf-8"
    )
    _write_figures(summary, pilot_output, phase1_summary_output)


def main() -> int:
    """Analyze the frozen pilot, write summaries and figures, and print the summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PILOT_OUTPUT)
    parser.add_argument(
        "--phase1-summary-output", type=Path, default=PHASE1_SUMMARY_OUTPUT
    )
    args = parser.parse_args()
    rows, summary = analyze_pilot()
    write_outputs(rows, summary, args.output, args.phase1_summary_output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
