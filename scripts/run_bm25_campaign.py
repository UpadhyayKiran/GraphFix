"""
Purpose:
    Run the readiness audit, freeze the cohort, or resume the full BM25 evaluation.

Input:
    A local snapshot, repository cache, results directory, campaign command, and BM25 settings.

Output:
    Readiness, cohort, checkpoint, failure, per-instance, and aggregate campaign artifacts.

Pipeline stage:
    Resumable full-cohort BM25 baseline execution.

When to run:
    Run this script when you need this operational check or stage.

Run example:
    python scripts/run_bm25_campaign.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from datasets import load_from_disk  # type: ignore[import-untyped]

from graphfix.datasets import LocBenchDataset, normalize_locbench_records
from graphfix.pipeline import BM25LocalizationPipeline, RetrievalConfig
from graphfix.pipeline.campaign import (
    EvaluationCohort,
    ReadinessRecord,
    freeze_cohort,
    readiness_counts,
    run_full_evaluation,
    run_readiness_audit,
)
from graphfix.repositories import RepositoryManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = PROJECT_ROOT / "data" / "raw" / "locbench_v1" / "dataset"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "cache" / "repositories"
DEFAULT_BM25_RESULTS = PROJECT_ROOT / "results" / "bm25_baseline"


def load_dataset(snapshot: Path) -> LocBenchDataset:
    """
    Purpose:
        Load and normalize the local dataset snapshot for campaign commands.

    Args:
        snapshot: The snapshot value used by this operation.

    Returns:
        A `LocBenchDataset` result produced by the operation.

    Pipeline role:
        Operational script support.
    """
    raw = load_from_disk(str(snapshot))
    return normalize_locbench_records(
        cast(Any, raw), raw_fields=cast(list[str], raw.column_names)
    )


def load_audit(path: Path) -> tuple[ReadinessRecord, ...]:
    """
    Purpose:
        Load a previously saved readiness audit.

    Args:
        path: The file or directory path to use.

    Returns:
        A `tuple[ReadinessRecord, ...]` result produced by the operation.

    Pipeline role:
        Operational script support.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(ReadinessRecord.model_validate(item) for item in payload)


def parser() -> argparse.ArgumentParser:
    """
    Purpose:
        Create the command-line parser for readiness, cohort, and evaluation campaign stages.

    Returns:
        A `argparse.ArgumentParser` result produced by the operation.

    Pipeline role:
        Operational script support.
    """
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("phase", choices=("audit", "cohort", "evaluate"))
    root.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    root.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    root.add_argument("--results-dir", type=Path, default=DEFAULT_BM25_RESULTS)
    root.add_argument("--retry-failures", action="store_true")
    return root


def main() -> int:
    """
    Purpose:
        Parse command-line arguments, validate numeric options, and run the selected handler.

    Returns:
        A `int` result produced by the operation.

    Pipeline role:
        Operational script support.
    """
    args = parser().parse_args()
    dataset = load_dataset(args.snapshot)
    manager = RepositoryManager(args.cache_root)
    if args.phase == "audit":
        records = run_readiness_audit(
            dataset, manager, args.results_dir, retry_failures=args.retry_failures
        )
        print(json.dumps(readiness_counts(records), indent=2))
        return 0
    records = load_audit(args.results_dir / "readiness_audit.json")
    if args.phase == "cohort":
        cohort = freeze_cohort(dataset, records, args.results_dir)
        print(f"included={len(cohort.included_ids)} excluded={len(cohort.excluded)}")
        return 0
    cohort = EvaluationCohort.model_validate_json(
        (args.results_dir / "evaluation_cohort.json").read_text(encoding="utf-8")
    )
    pipeline = BM25LocalizationPipeline(manager, config=RetrievalConfig())
    evaluations = run_full_evaluation(
        dataset,
        cohort,
        pipeline,
        args.results_dir,
        retry_failures=args.retry_failures,
    )
    print(f"completed={len(evaluations)} cohort={len(cohort.included_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
