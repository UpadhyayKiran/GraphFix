"""
Purpose:
    Compare localization methods on the first fixed benchmark cohort IDs.

Input:
    Local snapshot, frozen cohort file, method names, and a small limit.

Output:
    Per-instance JSON plus aggregate JSON/CSV under
    results/retrieval_comparison/pilot_evaluation.

Pipeline stage:
    Pilot evaluation; gold is read only after each localization result exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, cast

from datasets import load_from_disk  # type: ignore[import-untyped]

from graphfix.agent import OllamaConfig, OllamaTextGenerator, TextGenerator
from graphfix.datasets import normalize_locbench_records
from graphfix.evaluation import aggregate_metrics, evaluate_rankings, extract_gold_locations
from graphfix.pipeline import (
    AdaptiveLocalizationResult,
    AdaptiveNavigationConfig,
    AdaptiveNavigationPipeline,
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
OUTPUT = ROOT / "results" / "retrieval_comparison" / "pilot_evaluation"
ADAPTIVE_EXPERIMENTS = ROOT / "results" / "adaptive_navigation_experiments"
FIXED_ADAPTIVE_OUTPUT = ADAPTIVE_EXPERIMENTS / "fixed_pool"
EXPANDED_ADAPTIVE_OUTPUT = ADAPTIVE_EXPERIMENTS / "expanded_pool"
BASE_METHODS = ("bm25", "dense", "hybrid", "graph")
METHODS = (*BASE_METHODS, "adaptive")


def _cohort_path() -> Path:
    """Use the semantic cohort path, with an explicit read-only legacy fallback."""
    if COHORT.exists():
        return COHORT
    if LEGACY_COHORT.exists():
        print(f"Warning: using legacy cohort artifact read-only: {LEGACY_COHORT}")
        return LEGACY_COHORT
    raise FileNotFoundError(f"Cohort artifact not found: {COHORT}")


def _persist_then_evaluate(
    localization: LocalizationResult | AdvancedLocalizationResult | AdaptiveLocalizationResult,
    instance: Any,
    path: Path,
) -> Any:
    """Persist a gold-free result before crossing the GraphFixGold boundary."""
    save_result(localization, path)
    gold = extract_gold_locations(instance.gold)
    return evaluate_rankings(
        localization.file_rankings, localization.function_rankings, gold
    )


def run_pilot(
    *,
    limit: int,
    start_index: int = 0,
    methods: tuple[str, ...],
    output: Path,
    generator: TextGenerator | None = None,
    adaptive_config: AdaptiveNavigationConfig | None = None,
) -> dict[str, Any]:
    """Run methods over the first frozen cohort IDs and aggregate actual metrics."""
    raw = load_from_disk(str(SNAPSHOT))
    dataset = normalize_locbench_records(
        cast(Any, raw), raw_fields=cast(list[str], raw.column_names)
    )
    instances = {item.input.instance_id: item for item in dataset.instances}
    cohort_payload = json.loads(_cohort_path().read_text(encoding="utf-8"))
    instance_ids = tuple(
        cohort_payload["included_ids"][start_index : start_index + limit]
    )
    manager = RepositoryManager(CACHE)
    bm25_pipeline = BM25LocalizationPipeline(manager)
    advanced = AdvancedLocalizationPipeline(manager)
    adaptive = (
        AdaptiveNavigationPipeline(
            manager,
            generator,
            config=adaptive_config,
            hybrid_pipeline=advanced,
        )
        if generator is not None and adaptive_config is not None
        else None
    )
    summaries: list[dict[str, Any]] = []
    metrics_by_method: dict[str, list[Any]] = {method: [] for method in methods}
    runtime_by_method = {method: 0.0 for method in methods}
    per_instance: list[dict[str, Any]] = []
    preparation_rows: list[dict[str, Any]] = []

    # Instance-major order lets Dense, Hybrid, and Graph reuse the exact same
    # in-memory document embeddings before moving to another repository state.
    for instance_id in instance_ids:
        instance = instances[instance_id]
        preparation_started = time.perf_counter()
        try:
            advanced.localize(instance.input, method="dense")
            preparation_failure = None
        except Exception as error:
            preparation_failure = f"{type(error).__name__}: {error}"
        preparation_runtime = time.perf_counter() - preparation_started
        preparation_rows.append(
            {
                "instance_id": instance_id,
                "cold_end_to_end_seconds": preparation_runtime,
                "failure": preparation_failure,
            }
        )
        for method in methods:
            localization: (
                LocalizationResult
                | AdvancedLocalizationResult
                | AdaptiveLocalizationResult
            )
            started = time.perf_counter()
            try:
                if method == "adaptive":
                    if adaptive is None:
                        raise RuntimeError("Adaptive method requires a configured controller")
                    localization = adaptive.localize(instance.input)
                elif method == "bm25":
                    localization = bm25_pipeline.localize(instance.input)
                else:
                    localization = advanced.localize(
                        instance.input, method=cast(Any, method)
                    )
                runtime = time.perf_counter() - started
                metrics = _persist_then_evaluate(
                    localization,
                    instance,
                    output / method / f"{instance_id}.json",
                )
                metrics_by_method[method].append(metrics)
                runtime_by_method[method] += runtime
                row = {
                    "instance_id": instance_id,
                    "method": method,
                    "runtime_seconds": runtime,
                    "failure": None,
                    **metrics.model_dump(),
                }
                if method == "adaptive":
                    adaptive_result = cast(AdaptiveLocalizationResult, localization)
                    row["resources"] = {
                        "runtime_seconds": adaptive_result.runtime_seconds,
                        "controller_calls": (
                            adaptive_result.file_navigation.model_calls
                            + adaptive_result.function_navigation.model_calls
                        ),
                        "actions_taken": (
                            adaptive_result.file_navigation.actual_steps
                            + adaptive_result.function_navigation.actual_steps
                        ),
                        "graph_nodes_observed": len(
                            set(adaptive_result.file_navigation.observed_node_ids)
                            | set(adaptive_result.function_navigation.observed_node_ids)
                        ),
                        "file_stopping_reason": (
                            adaptive_result.file_navigation.stopping_reason
                        ),
                        "function_stopping_reason": (
                            adaptive_result.function_navigation.stopping_reason
                        ),
                        "retry_count": (
                            adaptive_result.file_navigation.retry_count
                            + adaptive_result.function_navigation.retry_count
                        ),
                    }
            except Exception as error:
                runtime = time.perf_counter() - started
                runtime_by_method[method] += runtime
                row = {
                    "instance_id": instance_id,
                    "method": method,
                    "runtime_seconds": runtime,
                    "failure": f"{type(error).__name__}: {error}",
                    "file": None,
                    "function": None,
                }
            per_instance.append(row)

    for method in methods:
        aggregate = aggregate_metrics(metrics_by_method[method]).model_dump()
        for level in ("file", "function"):
            summaries.append(
                {
                    "method": method,
                    "level": level,
                    "hit_at_1": aggregate[f"{level}_hit_at_1"],
                    "recall_at_5": aggregate[f"{level}_recall_at_5"],
                    "recall_at_10": aggregate[f"{level}_recall_at_10"],
                    "mrr": aggregate[f"{level}_mrr"],
                    "attempted": len(instance_ids),
                    "completed": len(metrics_by_method[method]),
                    "failed": len(instance_ids) - len(metrics_by_method[method]),
                    "runtime_seconds": runtime_by_method[method],
                    "average_runtime_seconds": (
                        runtime_by_method[method] / len(instance_ids)
                        if instance_ids
                        else 0.0
                    ),
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "cohort_ids": list(instance_ids),
        "timing_methodology": (
            "One cold Dense preparation pass per instance, followed by separately timed "
            "warm end-to-end BM25, Dense, Hybrid, and Graph runs."
        ),
        "cold_preparation_total_seconds": sum(
            row["cold_end_to_end_seconds"] for row in preparation_rows
        ),
        "cold_preparation_average_seconds": (
            sum(row["cold_end_to_end_seconds"] for row in preparation_rows)
            / len(preparation_rows)
        ),
        "warm_methods_total_seconds": sum(runtime_by_method.values()),
        "rows": summaries,
    }
    (output / "comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output / "per_instance.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_instance:
            handle.write(json.dumps(row) + "\n")
    with (output / "preparation.jsonl").open("w", encoding="utf-8") as handle:
        for row in preparation_rows:
            handle.write(json.dumps(row) + "\n")
    return payload


def main() -> int:
    """Parse a bounded pilot request and write its comparison artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--adaptive-output", type=Path)
    parser.add_argument("--ollama-model")
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--controller-timeout", type=float, default=120.0)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--candidate-pool", choices=("fixed", "expanded"), default="fixed"
    )
    parser.add_argument("--max-expanded-candidates", type=int, default=5)
    args = parser.parse_args()
    if args.limit < 1 or args.start_index < 0 or args.start_index + args.limit > 560:
        parser.error("--start-index and --limit must select within the 560-instance cohort")
    if args.max_expanded_candidates < 1:
        parser.error("--max-expanded-candidates must be positive")
    methods = tuple(args.method or BASE_METHODS)
    if "adaptive" in methods and len(methods) != 1:
        parser.error("Run adaptive separately so existing comparison artifacts are not overwritten")
    if methods != ("adaptive",) and args.candidate_pool != "fixed":
        parser.error("--candidate-pool applies only to the adaptive method")
    generator = None
    adaptive_config = None
    output = args.output
    if methods == ("adaptive",):
        if not args.ollama_model:
            parser.error("--ollama-model is required for the adaptive method")
        ollama = OllamaConfig(
            model=args.ollama_model,
            endpoint=args.ollama_endpoint,
            timeout_seconds=args.controller_timeout,
        )
        generator = OllamaTextGenerator(ollama)
        adaptive_config = AdaptiveNavigationConfig(
            provider="ollama",
            model=ollama.model,
            endpoint=ollama.endpoint,
            timeout_seconds=ollama.timeout_seconds,
            temperature=ollama.temperature,
            seed=ollama.seed,
            max_steps=args.max_steps,
            max_retries=args.max_retries,
            candidate_pool=args.candidate_pool,
            max_expanded_candidates=args.max_expanded_candidates,
        )
        output = args.adaptive_output or (
            EXPANDED_ADAPTIVE_OUTPUT
            if args.candidate_pool == "expanded"
            else FIXED_ADAPTIVE_OUTPUT
        )
    print(
        json.dumps(
            run_pilot(
                limit=args.limit,
                start_index=args.start_index,
                methods=methods,
                output=output,
                generator=generator,
                adaptive_config=adaptive_config,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
