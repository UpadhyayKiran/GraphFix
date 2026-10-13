"""Validate offline ten-stage coverage for the frozen 560-issue demo cohort."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, cast

from datasets import load_from_disk  # type: ignore[import-untyped]

from graphfix.datasets import GraphFixGold
from graphfix.evaluation import evaluate_rankings, extract_gold_locations
from graphfix.pipeline.models import AdaptiveLocalizationResult
from graphfix.retrieval import RankedCandidate

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "results" / "localization_benchmark" / "evaluation_cohort.json"
DATASET = ROOT / "data" / "raw" / "locbench_v1" / "dataset"
DETERMINISTIC = ROOT / "results" / "retrieval_comparison" / "full_evaluation"
ADAPTIVE_PARTITIONS = (
    ROOT / "results" / "adaptive_navigation_experiments" / "fixed_pool_100" / "adaptive",
    ROOT
    / "results"
    / "adaptive_navigation_experiments"
    / "fixed_pool_101_560"
    / "adaptive",
)
METHODS = ("bm25", "dense", "hybrid", "graph")


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def local_checkout(saved_path: str) -> Path:
    """Resolve a frozen checkout path against the current package when possible."""
    saved = Path(saved_path)
    try:
        index = saved.parts.index("repositories")
    except ValueError:
        return saved
    packaged = ROOT / "data" / "cache" / Path(*saved.parts[index:])
    return packaged if packaged.is_dir() else saved


def ranking(payload: dict[str, Any], level: str) -> tuple[RankedCandidate, ...]:
    """Validate a persisted deterministic ranking."""
    rows = cast(list[dict[str, Any]], payload[f"{level}_rankings"])
    return tuple(RankedCandidate.model_validate(row) for row in rows)


def main() -> None:
    """Fail fast unless every cohort issue supports the complete offline demo."""
    cohort = tuple(cast(list[str], read_json(COHORT)["included_ids"]))
    if len(cohort) != 560 or len(set(cohort)) != 560:
        raise ValueError("The cohort must contain exactly 560 unique ordered IDs")

    dataset = load_from_disk(str(DATASET))
    dataset_rows = {
        cast(str, row["instance_id"]): cast(dict[str, Any], row)
        for row in dataset
        if cast(str, row["instance_id"]) in set(cohort)
    }
    if set(dataset_rows) != set(cohort):
        raise ValueError("The dataset ID set does not match the frozen cohort")

    adaptive_paths: dict[str, Path] = {}
    for directory in ADAPTIVE_PARTITIONS:
        for path in directory.glob("*.json"):
            if path.stem in adaptive_paths:
                raise ValueError(f"Duplicate Adaptive artifact: {path.stem}")
            adaptive_paths[path.stem] = path
    if set(adaptive_paths) != set(cohort):
        raise ValueError("The definitive Adaptive ID set does not match the frozen cohort")

    for position, instance_id in enumerate(cohort, start=1):
        row = dataset_rows[instance_id]
        adaptive = AdaptiveLocalizationResult.model_validate_json(
            adaptive_paths[instance_id].read_text(encoding="utf-8")
        )
        if adaptive.adaptive_config.candidate_pool != "fixed":
            raise ValueError(f"Non-fixed Adaptive artifact: {instance_id}")
        if not adaptive.file_navigation.trace or not adaptive.function_navigation.trace:
            raise ValueError(f"Empty Adaptive trace: {instance_id}")
        checkout = local_checkout(adaptive.checkout_path)
        if not checkout.is_dir():
            raise FileNotFoundError(f"Checkout is absent: {instance_id}: {checkout}")
        head = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        expected = cast(str, row["base_commit"])
        if (
            adaptive.base_commit != expected
            or adaptive.checked_out_commit != expected
            or head != expected
        ):
            raise ValueError(f"Historical commit mismatch: {instance_id}")

        methods = {
            method: read_json(DETERMINISTIC / method / f"{instance_id}.json")
            for method in METHODS
        }
        for payload in methods.values():
            ranking(payload, "file")
            ranking(payload, "function")

        gold = GraphFixGold(
            patch=cast(str, row["patch"]),
            test_patch=cast(str, row["test_patch"]),
            labels=cast(list[str] | None, row["labels"]),
            edit_functions=cast(list[str], row["edit_functions"]),
            added_functions=cast(list[str], row["added_functions"]),
        )
        locations = extract_gold_locations(gold)
        hybrid = methods["hybrid"]
        evaluate_rankings(ranking(hybrid, "file"), ranking(hybrid, "function"), locations)
        evaluate_rankings(adaptive.file_rankings, adaptive.function_rankings, locations)
        if position % 100 == 0:
            print(f"Validated {position}/560")

    print("PHASE1_DEMO_COVERAGE: 560/560 PASS")


if __name__ == "__main__":
    main()
