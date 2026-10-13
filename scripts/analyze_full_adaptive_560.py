"""Audit and analyze the complete frozen 560-instance Adaptive campaign."""

# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from datasets import load_from_disk  # type: ignore[import-untyped]

from graphfix.datasets import normalize_locbench_records
from graphfix.evaluation import aggregate_metrics, evaluate_rankings, extract_gold_locations
from graphfix.pipeline import AdaptiveLocalizationResult

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "results" / "localization_benchmark" / "evaluation_cohort.json"
SNAPSHOT = ROOT / "data" / "raw" / "locbench_v1" / "dataset"
FIRST100 = ROOT / "results" / "adaptive_navigation_experiments" / "fixed_pool_100" / "adaptive"
CONTINUATION = (
    ROOT
    / "results"
    / "adaptive_navigation_experiments"
    / "fixed_pool_101_560"
    / "adaptive"
)
BASELINES = ROOT / "results" / "retrieval_comparison" / "full_evaluation"
OUTPUT = ROOT / "results" / "adaptive_navigation_experiments" / "fixed_pool_560_analysis"
EXPECTED_FIRST100 = {
    "file_hit_at_1": 0.420000,
    "file_recall_at_5": 0.336452,
    "file_recall_at_10": 0.413401,
    "file_mrr": 0.549329,
    "function_hit_at_1": 0.330000,
    "function_recall_at_5": 0.213093,
    "function_recall_at_10": 0.280361,
    "function_mrr": 0.405734,
}
METRICS = ("hit_at_1", "recall_at_5", "recall_at_10", "mrr")


def _outcome(before: int | None, after: int | None) -> str:
    before_value = before if before is not None else 10**9
    after_value = after if after is not None else 10**9
    if after_value < before_value:
        return "IMPROVED"
    if after_value > before_value:
        return "DEGRADED"
    return "UNCHANGED"


def _load_artifacts(
    expected_ids: list[str], directory: Path
) -> list[AdaptiveLocalizationResult]:
    paths = sorted(directory.glob("*.json"))
    expected = set(expected_ids)
    filenames = [path.stem for path in paths]
    if len(paths) != len(expected_ids):
        raise RuntimeError(f"{directory}: expected {len(expected_ids)} JSON files, found {len(paths)}")
    if set(filenames) != expected:
        raise RuntimeError(
            f"{directory}: missing={sorted(expected - set(filenames))}, "
            f"unexpected={sorted(set(filenames) - expected)}"
        )
    results_by_id: dict[str, AdaptiveLocalizationResult] = {}
    for path in paths:
        result = AdaptiveLocalizationResult.model_validate_json(path.read_text(encoding="utf-8"))
        if result.instance_id != path.stem:
            raise RuntimeError(f"Artifact ID mismatch: {path.name} contains {result.instance_id}")
        if result.instance_id in results_by_id:
            raise RuntimeError(f"Duplicate artifact ID: {result.instance_id}")
        if result.adaptive_config.candidate_pool != "fixed":
            raise RuntimeError(f"Non-fixed candidate pool: {path}")
        results_by_id[result.instance_id] = result
    return [results_by_id[instance_id] for instance_id in expected_ids]


def _efficiency(results: list[AdaptiveLocalizationResult]) -> dict[str, Any]:
    levels = [
        level
        for result in results
        for level in (result.file_navigation, result.function_navigation)
    ]
    repeated = 0
    new_evidence = 0
    for level in levels:
        seen_pairs: set[tuple[str, str | None]] = set()
        seen_evidence: set[tuple[str, str, str, str]] = set()
        for step in level.trace:
            pair = (step.action, step.node_id)
            repeated += int(pair in seen_pairs)
            seen_pairs.add(pair)
            additions = 0
            for node in step.observed_nodes:
                evidence = (node.node_id, node.relationship, node.direction, node.snippet)
                if evidence not in seen_evidence:
                    seen_evidence.add(evidence)
                    additions += 1
            new_evidence += int(additions > 0)
    accepted_steps = sum(level.actual_steps for level in levels)
    stopping = Counter(level.stopping_reason for level in levels)
    runtimes = [result.runtime_seconds for result in results]
    return {
        "issues": len(results),
        "episodes": len(levels),
        "accepted_steps": accepted_steps,
        "controller_calls": sum(level.model_calls for level in levels),
        "retries": sum(level.retry_count for level in levels),
        "validation_failures": sum(level.validation_failures for level in levels),
        "repeated_action_target_count": repeated,
        "repeated_action_target_rate": repeated / accepted_steps,
        "new_evidence_count": new_evidence,
        "new_evidence_rate": new_evidence / accepted_steps,
        "stopping": {
            reason: {"count": stopping[reason], "rate": stopping[reason] / len(levels)}
            for reason in ("explicit_stop", "invalid_response_stop", "max_steps")
        },
        "persisted_localization_runtime_seconds": sum(runtimes),
        "mean_runtime_seconds_per_issue": statistics.mean(runtimes),
        "median_runtime_seconds_per_issue": statistics.median(runtimes),
    }


def main() -> int:
    cohort_payload = json.loads(COHORT.read_text(encoding="utf-8"))
    cohort: list[str] = cohort_payload["included_ids"]
    if len(cohort) != 560 or len(set(cohort)) != 560:
        raise RuntimeError("Frozen cohort must contain exactly 560 unique IDs")

    first100 = _load_artifacts(cohort[:100], FIRST100)
    continuation = _load_artifacts(cohort[100:], CONTINUATION)
    results = first100 + continuation
    combined_ids = [result.instance_id for result in results]
    if combined_ids != cohort or len(set(combined_ids)) != 560:
        raise RuntimeError("Combined Adaptive artifacts do not exactly match cohort order")
    configs = {json.dumps(result.adaptive_config.model_dump(mode="json"), sort_keys=True) for result in results}
    if len(configs) != 1:
        raise RuntimeError("Adaptive artifacts contain multiple configurations")

    baseline_payload = json.loads((BASELINES / "comparison.json").read_text(encoding="utf-8"))
    if baseline_payload["cohort_ids"] != cohort:
        raise RuntimeError("Frozen deterministic baseline cohort/order differs")
    if any(row["completed"] != 560 or row["failed"] != 0 for row in baseline_payload["rows"]):
        raise RuntimeError("Frozen deterministic baseline is incomplete")

    raw = load_from_disk(str(SNAPSHOT))
    dataset = normalize_locbench_records(
        cast(Any, raw), raw_fields=cast(list[str], raw.column_names)
    )
    instances = {item.input.instance_id: item for item in dataset.instances}
    adaptive_metrics = []
    hybrid_metrics = []
    outcome_rows: list[dict[str, Any]] = []
    file_outcomes: Counter[str] = Counter()
    function_outcomes: Counter[str] = Counter()
    for result in results:
        gold = extract_gold_locations(instances[result.instance_id].gold)
        hybrid = evaluate_rankings(
            result.file_navigation.initial_ranking,
            result.function_navigation.initial_ranking,
            gold,
        )
        adaptive = evaluate_rankings(result.file_rankings, result.function_rankings, gold)
        hybrid_metrics.append(hybrid)
        adaptive_metrics.append(adaptive)
        file_outcome = _outcome(hybrid.file.first_relevant_rank, adaptive.file.first_relevant_rank)
        function_outcome = _outcome(
            hybrid.function.first_relevant_rank, adaptive.function.first_relevant_rank
        )
        file_outcomes[file_outcome] += 1
        function_outcomes[function_outcome] += 1
        outcome_rows.append(
            {
                "instance_id": result.instance_id,
                "hybrid_file_rank": hybrid.file.first_relevant_rank,
                "adaptive_file_rank": adaptive.file.first_relevant_rank,
                "file_outcome": file_outcome,
                "hybrid_function_rank": hybrid.function.first_relevant_rank,
                "adaptive_function_rank": adaptive.function.first_relevant_rank,
                "function_outcome": function_outcome,
            }
        )

    full_adaptive = aggregate_metrics(adaptive_metrics).model_dump()
    full_hybrid_from_artifacts = aggregate_metrics(hybrid_metrics).model_dump()
    first100_metrics = aggregate_metrics(adaptive_metrics[:100]).model_dump()
    regression = {
        key: {
            "actual": first100_metrics[key],
            "expected_rounded": expected,
            "within_1e-6": abs(first100_metrics[key] - expected) <= 1e-6,
        }
        for key, expected in EXPECTED_FIRST100.items()
    }
    if not all(item["within_1e-6"] for item in regression.values()):
        raise RuntimeError("First-100 Adaptive metrics failed the historical regression check")

    baseline_by_method: dict[str, dict[str, float]] = {}
    for row in baseline_payload["rows"]:
        method = "static_graph" if row["method"] == "graph" else row["method"]
        baseline_by_method.setdefault(method, {})
        for metric in METRICS:
            baseline_by_method[method][f"{row['level']}_{metric}"] = row[metric]
    baseline_hybrid = baseline_by_method["hybrid"]
    if any(abs(full_hybrid_from_artifacts[key] - baseline_hybrid[key]) > 1e-12 for key in baseline_hybrid):
        raise RuntimeError("Adaptive initial Hybrid rankings differ from frozen full-560 Hybrid metrics")
    comparison = [
        {"method": method, **baseline_by_method[method]}
        for method in ("bm25", "dense", "hybrid", "static_graph")
    ]
    comparison.append({"method": "adaptive_fixed", **{key: full_adaptive[key] for key in baseline_hybrid}})
    deltas = {key: full_adaptive[key] - baseline_hybrid[key] for key in baseline_hybrid}
    outcomes = {
        "definition": "Lower first relevant rank is improved; higher is degraded; equal (including both absent from Top-10) is unchanged. Combined sums the file and function classifications.",
        "file": {key.lower(): file_outcomes[key] for key in ("IMPROVED", "UNCHANGED", "DEGRADED")},
        "function": {key.lower(): function_outcomes[key] for key in ("IMPROVED", "UNCHANGED", "DEGRADED")},
        "combined": {
            key.lower(): file_outcomes[key] + function_outcomes[key]
            for key in ("IMPROVED", "UNCHANGED", "DEGRADED")
        },
    }
    efficiency = _efficiency(results)
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    summary = {
        "schema_version": 1,
        "generated_at": generated_at,
        "provenance": {
            "cohort": str(COHORT.relative_to(ROOT)),
            "cohort_size": 560,
            "first100_artifacts": str(FIRST100.relative_to(ROOT)),
            "continuation_artifacts": str(CONTINUATION.relative_to(ROOT)),
            "deterministic_baselines": str(BASELINES.relative_to(ROOT)),
            "candidate_pool_location": "adaptive_config.candidate_pool",
            "configuration": json.loads(next(iter(configs))),
            "evaluation_methodology": "Gold-free rankings loaded first; gold extracted post hoc with extract_gold_locations; evaluate_rankings and aggregate_metrics applied in frozen cohort order.",
        },
        "integrity": {
            "first100_count": 100,
            "continuation_count": 460,
            "combined_unique_count": 560,
            "missing_ids": [],
            "duplicate_ids": [],
            "unexpected_ids": [],
        },
        "first100_regression": regression,
        "adaptive_fixed_metrics": full_adaptive,
        "hybrid_to_adaptive_fixed_deltas": deltas,
        "outcomes": outcomes,
        "efficiency": efficiency,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "full560_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "full560_efficiency.json").write_text(json.dumps(efficiency, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT / "full560_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    with (OUTPUT / "full560_outcomes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outcome_rows[0]))
        writer.writeheader()
        writer.writerows(outcome_rows)

    markdown = _markdown(summary, comparison)
    (OUTPUT / "FULL560_ADAPTIVE_ANALYSIS.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _markdown(summary: dict[str, Any], comparison: list[dict[str, Any]]) -> str:
    metrics = summary["adaptive_fixed_metrics"]
    outcomes = summary["outcomes"]
    efficiency = summary["efficiency"]
    lines = [
        "# Full-560 Adaptive Fixed-Pool Analysis",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "All 560 artifacts exactly match the frozen cohort. The candidate-pool field is `adaptive_config.candidate_pool`; every artifact is `fixed`. Gold was loaded only for post-hoc evaluation.",
        "",
        "## Metrics",
        "",
        "| Method | File H@1 | File R@5 | File R@10 | File MRR | Function H@1 | Function R@5 | Function R@10 | Function MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append("| " + " | ".join([row["method"], *[f"{row[f'{level}_{metric}']:.6f}" for level in ("file", "function") for metric in METRICS]]) + " |")
    lines.extend(
        [
            "",
            "## Hybrid to Adaptive Fixed outcomes",
            "",
            outcomes["definition"],
            "",
            f"- File: {outcomes['file']}",
            f"- Function: {outcomes['function']}",
            f"- Combined: {outcomes['combined']}",
            "",
            "## Efficiency",
            "",
            f"- Episodes: {efficiency['episodes']}; accepted steps: {efficiency['accepted_steps']}; controller calls: {efficiency['controller_calls']}",
            f"- Repeated action-target rate: {efficiency['repeated_action_target_rate']:.6f}; new-evidence rate: {efficiency['new_evidence_rate']:.6f}",
            f"- Persisted runtime: {efficiency['persisted_localization_runtime_seconds']:.3f} seconds; mean: {efficiency['mean_runtime_seconds_per_issue']:.3f}; median: {efficiency['median_runtime_seconds_per_issue']:.3f}",
            "",
            "## Interpretation",
            "",
            "Adaptive Fixed is a prompted reranker of the frozen Hybrid Top-10 candidate pool. The full-cohort evidence should be described metric by metric: it changes first-rank quality and MRR while leaving the candidate-pool ceiling unchanged. Its controller calls, repeated actions, validation failures, and runtime are additional resource costs. These results support conclusions for this frozen Loc-Bench cohort and configuration only, not global superiority.",
            "",
            f"Adaptive metric record: `{json.dumps(metrics, sort_keys=True)}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
