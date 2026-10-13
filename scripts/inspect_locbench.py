"""
Purpose:
    Inspect the local Loc-Bench snapshot without printing patches or complete issue reports.

Input:
    The local dataset snapshot and an optional generated audit path.

Output:
    Safe counts, warnings, short row previews, and a JSON audit.

Pipeline stage:
    Dataset inspection before a run.

When to run:
    Run this script when you need this operational check or stage.

Run example:
    python scripts/inspect_locbench.py
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from graphfix.datasets import load_locbench

REPORT_PATH = Path("results/generated/locbench_dataset_audit.json")
PREVIEW_LENGTH = 120


def _preview(text: str) -> str:
    """
    Purpose:
        Flatten and shorten text for safe one-line dataset inspection.

    Args:
        text: Text to tokenize or normalize.

    Returns:
        A `str` result produced by the operation.

    Pipeline role:
        Operational script support.
    """
    compact = " ".join(text.split())
    if len(compact) <= PREVIEW_LENGTH:
        return compact
    return f"{compact[: PREVIEW_LENGTH - 1]}…"


def main() -> None:
    """Load the official test split, print safe statistics, and write an audit."""
    dataset = load_locbench()
    categories = Counter(instance.input.category for instance in dataset.instances)
    added_empty = sum(not instance.gold.added_functions for instance in dataset.instances)
    lengths = [len(instance.input.problem_statement) for instance in dataset.instances]
    anomaly_codes = Counter(anomaly.code for anomaly in dataset.anomalies)
    base_commit_warnings = [
        anomaly
        for anomaly in dataset.anomalies
        if anomaly.field == "base_commit"
    ]

    print(f"Records: {len(dataset)}")
    print(f"Raw fields: {', '.join(dataset.raw_fields)}")
    print("Category counts:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")
    print(f"Records with empty added_functions: {added_empty}")
    print(f"Records with non-empty added_functions: {len(dataset) - added_empty}")
    print(
        "Problem-statement length (characters): "
        f"min={min(lengths)}, mean={statistics.fmean(lengths):.1f}, "
        f"median={statistics.median(lengths):.1f}, max={max(lengths)}"
    )
    print(f"Base-commit format warnings: {len(base_commit_warnings)}")
    print(f"Validation warnings: {len(dataset.anomalies)}")
    if anomaly_codes:
        print("Warning counts:")
        for code, count in sorted(anomaly_codes.items()):
            print(f"  {code}: {count}")

    print("Safe summaries of first 5 records:")
    for instance in dataset.instances[:5]:
        visible = instance.input
        print(
            f"  {visible.instance_id} | {visible.repo} | {visible.base_commit[:12]} | "
            f"{visible.category} | edits={len(instance.gold.edit_functions)} | "
            f"additions={len(instance.gold.added_functions)}"
        )
        print(f"    problem: {_preview(visible.problem_statement)}")

    suspicious_ids = sorted({anomaly.instance_id for anomaly in dataset.anomalies})
    report: dict[str, Any] = {
        "dataset_name": dataset.dataset_name,
        "split": dataset.split,
        "record_count": len(dataset),
        "raw_fields": list(dataset.raw_fields),
        "category_counts": dict(sorted(categories.items())),
        "empty_added_functions": added_empty,
        "non_empty_added_functions": len(dataset) - added_empty,
        "problem_statement_length": {
            "minimum": min(lengths),
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "maximum": max(lengths),
        },
        "validation_warning_count": len(dataset.anomalies),
        "validation_warning_summary": dict(sorted(anomaly_codes.items())),
        "suspicious_instance_ids": suspicious_ids,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Audit report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
