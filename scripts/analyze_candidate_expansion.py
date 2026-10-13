"""Analyze fixed- versus expanded-pool adaptive rankings from saved traces."""

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
from graphfix.evaluation import aggregate_metrics, evaluate_rankings, extract_gold_locations
from graphfix.pipeline import AdaptiveLocalizationResult
from graphfix.retrieval import RankedCandidate

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "raw" / "locbench_v1" / "dataset"
COHORT = ROOT / "results" / "localization_benchmark" / "evaluation_cohort.json"
EXPANDED_DIR = (
    ROOT / "results" / "adaptive_navigation_experiments" / "expanded_pool" / "adaptive"
)
OUTPUT = ROOT / "results" / "adaptive_navigation_experiments" / "expanded_pool" / "analysis"
FROZEN_SUMMARY = (
    ROOT / "results" / "review2" / "pilot_summary" / "adaptive_pilot_summary.json"
)


def _fixed_rerank(
    initial: tuple[RankedCandidate, ...], observed_ids: tuple[str, ...]
) -> tuple[RankedCandidate, ...]:
    """Reproduce the fixed-pool rule from the same navigation evidence."""
    evidence = set(observed_ids)
    ordered = sorted(
        initial,
        key=lambda item: (
            not (
                item.document_id in evidence
                or f"file:{item.file_path}" in evidence
            ),
            item.rank,
        ),
    )
    return tuple(
        item.model_copy(update={"rank": rank})
        for rank, item in enumerate(ordered, start=1)
    )


def _outcome(before: int | None, after: int | None) -> str:
    """Classify a first-relevant-rank transition."""
    if before is None and after is None:
        return "UNCHANGED"
    if before is None or (after is not None and after < before):
        return "IMPROVED"
    if after is None or after > before:
        return "DEGRADED"
    return "UNCHANGED"


def _interpretation(before: int | None, after: int | None) -> str:
    """Label rescue, corruption, unnecessary navigation, or no effect."""
    outcome = _outcome(before, after)
    if before is None and after is not None:
        return "RESCUE"
    if before is not None and after is None:
        return "CORRUPTION"
    if outcome == "UNCHANGED" and before is not None and before <= 5:
        return "UNNECESSARY_NAVIGATION"
    return "NO_EFFECT" if outcome == "UNCHANGED" else outcome


def _trace_stats(level: Any) -> dict[str, Any]:
    """Measure accepted actions, novelty, repetition, and stopping from one trace."""
    seen_pairs: set[tuple[str, str | None]] = set()
    seen_evidence: set[tuple[str, str, str, str]] = set()
    repeated = 0
    new_evidence = 0
    unique_nodes: set[str] = set()
    for step in level.trace:
        pair = (step.action, step.node_id)
        repeated += int(pair in seen_pairs)
        seen_pairs.add(pair)
        additions = 0
        for node in step.observed_nodes:
            item = (node.node_id, node.relationship, node.direction, node.snippet)
            if item not in seen_evidence:
                seen_evidence.add(item)
                additions += 1
            unique_nodes.add(node.node_id)
        new_evidence += int(additions > 0)
    return {
        "steps": level.actual_steps,
        "model_calls": level.model_calls,
        "retries": level.retry_count,
        "validation_failures": level.validation_failures,
        "unique_nodes": len(unique_nodes),
        "repeated_action_targets": repeated,
        "new_evidence_steps": new_evidence,
        "stopping_reason": level.stopping_reason,
    }


def _aggregate_resources(results: list[AdaptiveLocalizationResult]) -> dict[str, Any]:
    """Aggregate resource use from persisted file/function traces."""
    stats = [
        _trace_stats(level)
        for result in results
        for level in (result.file_navigation, result.function_navigation)
    ]
    steps = sum(item["steps"] for item in stats)
    repeated = sum(item["repeated_action_targets"] for item in stats)
    new_evidence = sum(item["new_evidence_steps"] for item in stats)
    runtimes = [result.runtime_seconds for result in results]
    stops = Counter(item["stopping_reason"] for item in stats)
    episodes = len(stats)
    return {
        "episodes": episodes,
        "accepted_steps": steps,
        "model_calls": sum(item["model_calls"] for item in stats),
        "retries": sum(item["retries"] for item in stats),
        "validation_failures": sum(item["validation_failures"] for item in stats),
        "unique_node_observations": sum(item["unique_nodes"] for item in stats),
        "repeated_action_targets": repeated,
        "repetition_rate": repeated / steps if steps else 0.0,
        "new_evidence_steps": new_evidence,
        "new_evidence_rate": new_evidence / steps if steps else 0.0,
        "no_new_evidence_fraction": (steps - new_evidence) / steps if steps else 0.0,
        "stopping_reasons": dict(sorted(stops.items())),
        "explicit_stop_rate": stops["explicit_stop"] / episodes if episodes else 0.0,
        "invalid_response_stop_rate": (
            stops["invalid_response_stop"] / episodes if episodes else 0.0
        ),
        "max_step_rate": stops["max_steps"] / episodes if episodes else 0.0,
        "runtime_seconds": {
            "total": sum(runtimes),
            "mean": statistics.mean(runtimes),
            "median": statistics.median(runtimes),
        },
    }


def analyze(limit: int, expanded_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply gold only after loading saved localization results and counterfactuals."""
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))["included_ids"][:limit]
    raw = load_from_disk(str(SNAPSHOT))
    dataset = normalize_locbench_records(
        cast(Any, raw), raw_fields=cast(list[str], raw.column_names)
    )
    instances = {item.input.instance_id: item for item in dataset.instances}
    results = [
        AdaptiveLocalizationResult.model_validate_json(
            (expanded_dir / f"{instance_id}.json").read_text(encoding="utf-8")
        )
        for instance_id in cohort
    ]
    hybrid_metrics = []
    fixed_metrics = []
    expanded_metrics = []
    rows: list[dict[str, Any]] = []
    for result in results:
        fixed_files = _fixed_rerank(
            result.file_navigation.initial_ranking,
            result.file_navigation.observed_node_ids,
        )
        fixed_functions = _fixed_rerank(
            result.function_navigation.initial_ranking,
            result.function_navigation.observed_node_ids,
        )
        gold = extract_gold_locations(instances[result.instance_id].gold)
        hybrid = evaluate_rankings(
            result.file_navigation.initial_ranking,
            result.function_navigation.initial_ranking,
            gold,
        )
        fixed = evaluate_rankings(fixed_files, fixed_functions, gold)
        expanded = evaluate_rankings(result.file_rankings, result.function_rankings, gold)
        hybrid_metrics.append(hybrid)
        fixed_metrics.append(fixed)
        expanded_metrics.append(expanded)
        rows.append(
            {
                "instance_id": result.instance_id,
                "issue": result.issue,
                "hybrid_file_rank": hybrid.file.first_relevant_rank,
                "fixed_file_rank": fixed.file.first_relevant_rank,
                "expanded_file_rank": expanded.file.first_relevant_rank,
                "hybrid_function_rank": hybrid.function.first_relevant_rank,
                "fixed_function_rank": fixed.function.first_relevant_rank,
                "expanded_function_rank": expanded.function.first_relevant_rank,
                "file_expansion_outcome": _outcome(
                    fixed.file.first_relevant_rank, expanded.file.first_relevant_rank
                ),
                "function_expansion_outcome": _outcome(
                    fixed.function.first_relevant_rank,
                    expanded.function.first_relevant_rank,
                ),
                "file_hybrid_to_expanded": _outcome(
                    hybrid.file.first_relevant_rank, expanded.file.first_relevant_rank
                ),
                "function_hybrid_to_expanded": _outcome(
                    hybrid.function.first_relevant_rank,
                    expanded.function.first_relevant_rank,
                ),
                "file_interpretation": _interpretation(
                    hybrid.file.first_relevant_rank, expanded.file.first_relevant_rank
                ),
                "function_interpretation": _interpretation(
                    hybrid.function.first_relevant_rank,
                    expanded.function.first_relevant_rank,
                ),
                "expanded_file_candidate_ids": list(
                    result.file_navigation.expanded_candidate_ids
                ),
                "expanded_function_candidate_ids": list(
                    result.function_navigation.expanded_candidate_ids
                ),
                "hybrid_file_top10": [
                    item.document_id for item in result.file_navigation.initial_ranking
                ],
                "fixed_file_top10": [item.document_id for item in fixed_files],
                "expanded_file_top10": [item.document_id for item in result.file_rankings],
                "hybrid_function_top10": [
                    item.document_id for item in result.function_navigation.initial_ranking
                ],
                "fixed_function_top10": [item.document_id for item in fixed_functions],
                "expanded_function_top10": [
                    item.document_id for item in result.function_rankings
                ],
                "gold_files": list(gold.files),
                "gold_functions": list(gold.functions),
                "file_trace": [step.model_dump() for step in result.file_navigation.trace],
                "function_trace": [
                    step.model_dump() for step in result.function_navigation.trace
                ],
            }
        )

    metrics = {
        "hybrid": aggregate_metrics(hybrid_metrics).model_dump(),
        "adaptive_fixed_same_trace": aggregate_metrics(fixed_metrics).model_dump(),
        "adaptive_expanded": aggregate_metrics(expanded_metrics).model_dump(),
    }
    expansion_outcomes = Counter(
        row[key]
        for row in rows
        for key in ("file_expansion_outcome", "function_expansion_outcome")
    )
    hybrid_outcomes = Counter(
        row[key]
        for row in rows
        for key in ("file_hybrid_to_expanded", "function_hybrid_to_expanded")
    )
    interpretations = Counter(
        row[key]
        for row in rows
        for key in ("file_interpretation", "function_interpretation")
    )
    summary = {
        "schema_version": 1,
        "cohort_size": len(rows),
        "configuration": results[0].adaptive_config.model_dump(),
        "metrics": metrics,
        "expanded_vs_fixed_same_trace": dict(sorted(expansion_outcomes.items())),
        "expanded_vs_hybrid": dict(sorted(hybrid_outcomes.items())),
        "interpretations": {
            key: interpretations.get(key, 0)
            for key in (
                "RESCUE",
                "CORRUPTION",
                "UNNECESSARY_NAVIGATION",
                "NO_EFFECT",
                "IMPROVED",
                "DEGRADED",
            )
        },
        "candidate_expansion": {
            "file_candidates_admitted": sum(
                len(row["expanded_file_candidate_ids"]) for row in rows
            ),
            "function_candidates_admitted": sum(
                len(row["expanded_function_candidate_ids"]) for row in rows
            ),
            "instances_with_expansion": sum(
                bool(row["expanded_file_candidate_ids"])
                or bool(row["expanded_function_candidate_ids"])
                for row in rows
            ),
        },
        "resources": _aggregate_resources(results),
        "historical_fixed_pool_summary": json.loads(
            FROZEN_SUMMARY.read_text(encoding="utf-8")
        ),
    }
    return rows, summary


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    """Persist reproducible post-hoc analysis without altering localization traces."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "per_instance.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    metric_rows = []
    for method, values in summary["metrics"].items():
        for level in ("file", "function"):
            metric_rows.append(
                {
                    "method": method,
                    "level": level,
                    **{
                        name: values[f"{level}_{name}"]
                        for name in ("hit_at_1", "recall_at_5", "recall_at_10", "mrr")
                    },
                }
            )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    vllm = next(row for row in rows if row["instance_id"] == "vllm-project__vllm-9390")
    (output / "vllm_9390_before_after.json").write_text(
        json.dumps(vllm, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    """Analyze one deterministic expanded-pool cohort from disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--expanded-dir", type=Path, default=EXPANDED_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows, summary = analyze(args.limit, args.expanded_dir)
    write_outputs(rows, summary, args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
