"""Read-only verification of the complete BM25/Dense/Hybrid/Static Graph comparison."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

METHODS = ("bm25", "dense", "hybrid", "graph")
METRICS = ("hit_at_1", "recall_at_5", "recall_at_10", "mrr")


def _comparison_dir(primary: Path, legacy: Path) -> Path:
    """Use the semantic directory, with an explicit read-only legacy fallback."""
    if primary.exists():
        return primary
    if legacy.exists():
        print(f"Warning: using legacy comparison artifacts read-only: {legacy}")
        return legacy
    raise FileNotFoundError(f"Comparison artifacts not found: {primary}")


def verify_comparison(results_dir: Path) -> dict[str, Any]:
    """Check row/file counts, failures, ID sets, and persisted aggregate metrics."""
    rows = [
        cast(dict[str, Any], json.loads(line))
        for line in (results_dir / "per_instance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    comparison = cast(
        dict[str, Any],
        json.loads((results_dir / "comparison.json").read_text(encoding="utf-8")),
    )
    cohort_ids = set(map(str, comparison["cohort_ids"]))
    by_method = Counter(str(row["method"]) for row in rows)
    method_file_counts = {
        method: len(tuple((results_dir / method).glob("*.json"))) for method in METHODS
    }
    method_id_sets = {
        method: {
            path.stem for path in (results_dir / method).glob("*.json")
        }
        for method in METHODS
    }
    aggregate_differences: dict[str, float] = {}
    for summary in cast(list[dict[str, Any]], comparison["rows"]):
        method = str(summary["method"])
        level = str(summary["level"])
        successful = [
            row
            for row in rows
            if row["method"] == method and row.get("failure") is None
        ]
        for metric in METRICS:
            recomputed = sum(float(row[level][metric]) for row in successful) / len(
                successful
            )
            key = f"{method}.{level}.{metric}"
            aggregate_differences[key] = recomputed - float(summary[metric])

    failures = sum(row.get("failure") is not None for row in rows)
    report: dict[str, Any] = {
        "rows": len(rows),
        "rows_by_method": dict(sorted(by_method.items())),
        "method_file_counts": method_file_counts,
        "failures": failures,
        "cohort_size": len(cohort_ids),
        "method_instance_ids_match_cohort": {
            method: ids == cohort_ids for method, ids in method_id_sets.items()
        },
        "aggregate_metric_differences": aggregate_differences,
    }
    report["verified"] = (
        len(rows) == 2240
        and failures == 0
        and len(cohort_ids) == 560
        and all(by_method[method] == 560 for method in METHODS)
        and all(method_file_counts[method] == 560 for method in METHODS)
        and all(ids == cohort_ids for ids in method_id_sets.values())
        and all(abs(value) <= 1e-12 for value in aggregate_differences.values())
    )
    return report


def main() -> int:
    """Print verification JSON and fail if any invariant is violated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/retrieval_comparison/full_evaluation"),
    )
    args = parser.parse_args()
    results_dir = _comparison_dir(args.results_dir, Path("results/review2/full"))
    report = verify_comparison(results_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
