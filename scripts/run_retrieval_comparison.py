"""
Purpose:
    Run a resumable retrieval comparison over the full frozen benchmark cohort.

Input:
    Local Loc-Bench snapshot, frozen cohort IDs, selected methods, and a limit.

Output:
    Incremental method results, per-instance metrics, and aggregate comparison files
    under results/retrieval_comparison/full_evaluation.

Pipeline stage:
    Full retrieval evaluation. Gold is accessed only after localization completes.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, cast

from datasets import load_from_disk  # type: ignore[import-untyped]

from graphfix.datasets import normalize_locbench_records
from graphfix.evaluation import evaluate_rankings, extract_gold_locations
from graphfix.pipeline import (
    AdvancedLocalizationPipeline,
    AdvancedLocalizationResult,
    BM25LocalizationPipeline,
    LocalizationResult,
    save_result,
)
from graphfix.repositories import RepositoryManager

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "raw" / "locbench_v1" / "dataset"
COHORT = ROOT / "results" / "localization_benchmark" / "evaluation_cohort.json"
LEGACY_COHORT = ROOT / "results" / "review1" / "evaluation_cohort.json"
CACHE = ROOT / "data" / "cache" / "repositories"
OUTPUT = ROOT / "results" / "retrieval_comparison" / "full_evaluation"
METHODS = ("bm25", "dense", "hybrid", "graph")


def _cohort_path() -> Path:
    """Use the semantic cohort path, with an explicit read-only legacy fallback."""
    if COHORT.exists():
        return COHORT
    if LEGACY_COHORT.exists():
        print(f"Warning: using legacy cohort artifact read-only: {LEGACY_COHORT}")
        return LEGACY_COHORT
    raise FileNotFoundError(f"Cohort artifact not found: {COHORT}")


def _load_checkpoint(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load previously saved per-instance rows for resume support."""
    if not path.exists():
        return {}

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = cast(dict[str, Any], json.loads(line))
        except json.JSONDecodeError:
            print(f"Warning: ignoring invalid checkpoint line {line_number}")
            continue
        rows[(str(row["instance_id"]), str(row["method"]))] = row
    return rows


def _save_checkpoint(
    path: Path,
    rows: dict[tuple[str, str], dict[str, Any]],
    instance_ids: tuple[str, ...],
) -> None:
    """Save current per-instance rows atomically after each method finishes."""
    instance_order = {
        instance_id: index for index, instance_id in enumerate(instance_ids)
    }
    method_order = {method: index for index, method in enumerate(METHODS)}
    ordered_rows = sorted(
        rows.values(),
        key=lambda row: (
            instance_order.get(str(row["instance_id"]), len(instance_order)),
            method_order.get(str(row["method"]), len(method_order)),
        ),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered_rows:
            handle.write(json.dumps(row) + "\n")
    temporary.replace(path)


def _method_complete(
    *,
    instance_id: str,
    method: str,
    output: Path,
    rows: dict[tuple[str, str], dict[str, Any]],
) -> bool:
    """Return True only for a successful row with a saved localization file."""
    row = rows.get((instance_id, method))
    if row is None or row.get("failure") is not None:
        return False
    return (output / method / f"{instance_id}.json").exists()


def _metric_average(
    rows: list[dict[str, Any]], level: str, metric: str
) -> float:
    """Calculate one macro-averaged metric over successful rows."""
    values = [
        float(cast(dict[str, Any], row[level])[metric])
        for row in rows
        if row.get("failure") is None and isinstance(row.get(level), dict)
    ]
    return sum(values) / len(values) if values else 0.0


def _write_comparison(
    *,
    output: Path,
    rows: dict[tuple[str, str], dict[str, Any]],
    instance_ids: tuple[str, ...],
    methods: tuple[str, ...],
) -> dict[str, Any]:
    """Rebuild aggregate metrics from the current checkpoint rows."""
    summaries: list[dict[str, Any]] = []

    for method in methods:
        method_rows = [
            row
            for (instance_id, row_method), row in rows.items()
            if instance_id in instance_ids and row_method == method
        ]
        successful_rows = [
            row for row in method_rows if row.get("failure") is None
        ]
        runtime_seconds = sum(
            float(row.get("runtime_seconds", 0.0)) for row in method_rows
        )

        for level in ("file", "function"):
            summaries.append(
                {
                    "method": method,
                    "level": level,
                    "hit_at_1": _metric_average(
                        successful_rows, level, "hit_at_1"
                    ),
                    "recall_at_5": _metric_average(
                        successful_rows, level, "recall_at_5"
                    ),
                    "recall_at_10": _metric_average(
                        successful_rows, level, "recall_at_10"
                    ),
                    "mrr": _metric_average(successful_rows, level, "mrr"),
                    "target": len(instance_ids),
                    "attempted": len(method_rows),
                    "completed": len(successful_rows),
                    "failed": sum(
                        row.get("failure") is not None for row in method_rows
                    ),
                    "pending": len(instance_ids) - len(method_rows),
                    "runtime_seconds": runtime_seconds,
                    "average_runtime_seconds": (
                        runtime_seconds / len(method_rows)
                        if method_rows
                        else 0.0
                    ),
                }
            )

    payload = {
        "cohort_ids": list(instance_ids),
        "timing_methodology": (
            "Dense preparation is run before unfinished advanced methods so the "
            "timed Dense, Hybrid, and Graph executions use prepared/cached embeddings."
        ),
        "rows": summaries,
    }

    (output / "comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    return payload


def run_full(
    *, limit: int, methods: tuple[str, ...], output: Path
) -> dict[str, Any]:
    """Run or resume the selected methods over the frozen cohort."""
    raw = load_from_disk(str(SNAPSHOT))
    dataset = normalize_locbench_records(
        cast(Any, raw), raw_fields=cast(list[str], raw.column_names)
    )
    instances = {item.input.instance_id: item for item in dataset.instances}
    cohort_payload = json.loads(_cohort_path().read_text(encoding="utf-8"))
    instance_ids = tuple(cohort_payload["included_ids"][:limit])

    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "per_instance.jsonl"
    rows = _load_checkpoint(checkpoint_path)

    manager = RepositoryManager(CACHE)
    bm25_pipeline = BM25LocalizationPipeline(manager)
    advanced = AdvancedLocalizationPipeline(manager)

    for index, instance_id in enumerate(instance_ids, start=1):
        instance = instances[instance_id]
        incomplete_methods = [
            method
            for method in methods
            if not _method_complete(
                instance_id=instance_id,
                method=method,
                output=output,
                rows=rows,
            )
        ]

        if not incomplete_methods:
            print(
                f"[{index}/{len(instance_ids)}] "
                f"{instance_id}: all methods SKIPPED"
            )
            continue

        # Prepare/load document embeddings once before timing unfinished advanced methods.
        if any(
            method in {"dense", "hybrid", "graph"}
            for method in incomplete_methods
        ):
            print(
                f"[{index}/{len(instance_ids)}] "
                f"{instance_id}: preparing embeddings"
            )
            try:
                advanced.localize(instance.input, method="dense")
            except Exception as error:
                # Individual method runs below still record their own failures.
                print(
                    f"[{index}/{len(instance_ids)}] "
                    f"{instance_id}: preparation warning - "
                    f"{type(error).__name__}: {error}"
                )

        for method in methods:
            if _method_complete(
                instance_id=instance_id,
                method=method,
                output=output,
                rows=rows,
            ):
                print(
                    f"[{index}/{len(instance_ids)}] "
                    f"{instance_id} {method}: SKIP"
                )
                continue

            localization: LocalizationResult | AdvancedLocalizationResult
            started = time.perf_counter()

            try:
                if method == "bm25":
                    localization = bm25_pipeline.localize(instance.input)
                else:
                    localization = advanced.localize(
                        instance.input, method=cast(Any, method)
                    )

                runtime = time.perf_counter() - started

                # Save localization before using benchmark gold.
                save_result(
                    localization,
                    output / method / f"{instance_id}.json",
                )

                gold = extract_gold_locations(instance.gold)
                metrics = evaluate_rankings(
                    localization.file_rankings,
                    localization.function_rankings,
                    gold,
                )

                row = {
                    "instance_id": instance_id,
                    "method": method,
                    "runtime_seconds": runtime,
                    "failure": None,
                    **metrics.model_dump(),
                }
                print(
                    f"[{index}/{len(instance_ids)}] "
                    f"{instance_id} {method}: "
                    f"COMPLETE ({runtime:.3f}s)"
                )
            except Exception as error:
                runtime = time.perf_counter() - started
                row = {
                    "instance_id": instance_id,
                    "method": method,
                    "runtime_seconds": runtime,
                    "failure": f"{type(error).__name__}: {error}",
                    "file": None,
                    "function": None,
                }
                print(
                    f"[{index}/{len(instance_ids)}] "
                    f"{instance_id} {method}: "
                    f"FAILED - {row['failure']}"
                )

            rows[(instance_id, method)] = row

            # Save immediately after every method so a restart can skip it.
            _save_checkpoint(checkpoint_path, rows, instance_ids)

        # Keep aggregate files current after every completed instance.
        _write_comparison(
            output=output,
            rows=rows,
            instance_ids=instance_ids,
            methods=methods,
        )

    return _write_comparison(
        output=output,
        rows=rows,
        instance_ids=instance_ids,
        methods=methods,
    )


def main() -> int:
    """Parse the full-campaign request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=560)
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if not 1 <= args.limit <= 560:
        parser.error("--limit must be between 1 and 560")

    methods = tuple(args.method or METHODS)
    payload = run_full(
        limit=args.limit,
        methods=methods,
        output=args.output,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
