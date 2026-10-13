"""Analyze a saved fixed-pool Adaptive Navigation cohort post hoc."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, cast

from datasets import load_from_disk  # type: ignore[import-untyped]

from graphfix.datasets import normalize_locbench_records
from graphfix.evaluation import aggregate_metrics, evaluate_rankings, extract_gold_locations
from graphfix.pipeline import AdaptiveLocalizationResult

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "results" / "localization_benchmark" / "evaluation_cohort.json"
SNAPSHOT = ROOT / "data" / "raw" / "locbench_v1" / "dataset"


def _outcome(before: int | None, after: int | None) -> str:
    """Classify first-relevant-rank movement, treating absence as worse than any rank."""
    before_value = before if before is not None else 10**9
    after_value = after if after is not None else 10**9
    if after_value < before_value:
        return "IMPROVED"
    if after_value > before_value:
        return "DEGRADED"
    return "UNCHANGED"


def _resources(results: list[AdaptiveLocalizationResult]) -> dict[str, Any]:
    """Aggregate auditable controller and runtime measurements."""
    levels = [
        level
        for result in results
        for level in (result.file_navigation, result.function_navigation)
    ]
    repeated = 0
    new_evidence = 0
    unique_nodes = 0
    for level in levels:
        seen_pairs: set[tuple[str, str | None]] = set()
        seen_evidence: set[tuple[str, str, str, str]] = set()
        observed: set[str] = set()
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
                observed.add(node.node_id)
            new_evidence += int(additions > 0)
        unique_nodes += len(observed)
    step_count = sum(level.actual_steps for level in levels)
    runtimes = [result.runtime_seconds for result in results]
    stopping = Counter(level.stopping_reason for level in levels)
    return {
        "episodes": len(levels),
        "accepted_steps": step_count,
        "model_calls": sum(level.model_calls for level in levels),
        "retries": sum(level.retry_count for level in levels),
        "validation_failures": sum(level.validation_failures for level in levels),
        "unique_node_observations": unique_nodes,
        "repeated_action_targets": repeated,
        "repetition_rate": repeated / step_count if step_count else 0.0,
        "new_evidence_steps": new_evidence,
        "new_evidence_rate": new_evidence / step_count if step_count else 0.0,
        "no_new_evidence_fraction": (
            (step_count - new_evidence) / step_count if step_count else 0.0
        ),
        "stopping_reasons": dict(sorted(stopping.items())),
        "runtime_seconds": {
            "total": sum(runtimes),
            "mean": statistics.mean(runtimes),
            "median": statistics.median(runtimes),
        },
    }


def analyze(results_dir: Path, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load gold-free artifacts first, then evaluate the deterministic cohort prefix."""
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))["included_ids"][:limit]
    results = [
        AdaptiveLocalizationResult.model_validate_json(
            (results_dir / f"{instance_id}.json").read_text(encoding="utf-8")
        )
        for instance_id in cohort
    ]
    if any(result.adaptive_config.candidate_pool != "fixed" for result in results):
        raise ValueError("All input artifacts must use candidate_pool='fixed'")

    raw = load_from_disk(str(SNAPSHOT))
    dataset = normalize_locbench_records(
        cast(Any, raw), raw_fields=cast(list[str], raw.column_names)
    )
    instances = {item.input.instance_id: item for item in dataset.instances}
    hybrid_metrics = []
    adaptive_metrics = []
    rows: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()
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
        file_outcome = _outcome(
            hybrid.file.first_relevant_rank, adaptive.file.first_relevant_rank
        )
        function_outcome = _outcome(
            hybrid.function.first_relevant_rank, adaptive.function.first_relevant_rank
        )
        outcomes.update((file_outcome, function_outcome))
        rows.append(
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

    summary = {
        "schema_version": 1,
        "cohort_size": len(results),
        "configuration": results[0].adaptive_config.model_dump(),
        "metrics": {
            "hybrid": aggregate_metrics(hybrid_metrics).model_dump(),
            "adaptive_fixed": aggregate_metrics(adaptive_metrics).model_dump(),
        },
        "rank_outcomes": {
            key: outcomes.get(key, 0) for key in ("IMPROVED", "UNCHANGED", "DEGRADED")
        },
        "resources": _resources(results),
    }
    return rows, summary


def main() -> int:
    """Analyze and persist one fixed-pool cohort summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = analyze(args.results_dir, args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "per_instance.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
