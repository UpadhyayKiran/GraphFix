"""
Purpose:
    Independently verify the frozen 560-instance BM25 baseline artifacts.

Input:
    The frozen BM25 result directory.

Output:
    Cross-check counts, recomputed metrics, and a verified true/false summary.

Pipeline stage:
    Post-run evidence verification without changing benchmark files.

When to run:
    Run this script when you need this operational check or stage.

Run example:
    python scripts/verify_bm25_results.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRIC_FIELDS = (
    "file_hit_at_1",
    "file_recall_at_5",
    "file_recall_at_10",
    "file_mrr",
    "function_hit_at_1",
    "function_recall_at_5",
    "function_recall_at_10",
    "function_mrr",
)


def _artifact(primary: Path, legacy: Path) -> Path:
    """Resolve one semantic artifact, with an explicit read-only legacy fallback."""
    if primary.exists():
        return primary
    if legacy.exists():
        print(f"Warning: using legacy artifact read-only: {legacy}")
        return legacy
    raise FileNotFoundError(f"Artifact not found: {primary}")


def _results_dir(primary: Path, legacy: Path) -> Path:
    """Use the semantic result directory, with an explicit read-only legacy fallback."""
    if primary.exists():
        return primary
    if legacy.exists():
        print(f"Warning: using legacy BM25 artifacts read-only: {legacy}")
        return legacy
    raise FileNotFoundError(f"BM25 result directory not found: {primary}")


def verify_results(results_dir: Path, cohort_path: Path) -> dict[str, Any]:
    """Recompute macro metrics and cross-check cohort, runs, results, and failures."""
    instance_dir = _artifact(
        results_dir / "localization_results", results_dir / "bm25_full_instances"
    )
    paths = sorted(instance_dir.glob("*.json"))
    evaluations = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    persisted = json.loads(
        _artifact(results_dir / "metrics.json", results_dir / "bm25_full_metrics.json")
        .read_text(encoding="utf-8")
    )
    cohort = json.loads(
        cohort_path.read_text(encoding="utf-8")
    )
    run_records = [
        json.loads(line)
        for line in _artifact(
            results_dir / "run_records.jsonl", results_dir / "bm25_full_run_records.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    localization_result_lines = sum(
        bool(line.strip())
        for line in _artifact(
            results_dir / "localization_results.jsonl",
            results_dir / "bm25_full_predictions.jsonl",
        )
        .read_text(encoding="utf-8")
        .splitlines()
    )
    failures = sum(
        bool(line.strip())
        for line in (results_dir / "failures.jsonl").read_text(encoding="utf-8").splitlines()
    )
    count = len(evaluations)
    recomputed = {
        field: sum(
            float(item["metrics"][level][metric])
            for item in evaluations
        ) / count
        for field in METRIC_FIELDS
        for level, metric in [
            ("function", field.removeprefix("function_"))
            if field.startswith("function_")
            else ("file", field.removeprefix("file_"))
        ]
    }
    persisted_metrics = persisted["metrics"]
    differences = {
        field: recomputed[field] - float(persisted_metrics[field])
        for field in METRIC_FIELDS
    }
    cohort_ids = set(cohort["included_ids"])
    result_ids = {item["localization"]["instance_id"] for item in evaluations}
    report = {
        "result_files": count,
        "cohort_size": len(cohort["included_ids"]),
        "completed": persisted["completed"],
        "run_records": len(run_records),
        "successful_run_records": sum(item["status"] == "success" for item in run_records),
        "localization_result_lines": localization_result_lines,
        "failure_lines": failures,
        "persisted_runtime_failures": persisted["runtime_failures"],
        "cohort_matches_results": cohort_ids == result_ids,
        "persisted_metrics": {field: persisted_metrics[field] for field in METRIC_FIELDS},
        "recomputed_metrics": recomputed,
        "metric_differences": differences,
    }
    report["verified"] = (
        count == 560
        and report["cohort_size"] == 560
        and report["completed"] == 560
        and report["successful_run_records"] == 560
        and localization_result_lines == 560
        and failures == 0
        and report["persisted_runtime_failures"] == 0
        and report["cohort_matches_results"]
        and all(abs(value) <= 1e-12 for value in differences.values())
    )
    return report


def main() -> int:
    """Print verification JSON and return a failing status on any discrepancy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/bm25_baseline"),
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("results/localization_benchmark/evaluation_cohort.json"),
    )
    args = parser.parse_args()
    cohort_path = _artifact(
        args.cohort, Path("results/review1/evaluation_cohort.json")
    )
    results_dir = _results_dir(args.results_dir, Path("results/review1"))
    report = verify_results(results_dir, cohort_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
