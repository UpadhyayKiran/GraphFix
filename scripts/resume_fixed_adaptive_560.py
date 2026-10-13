"""Resume the frozen fixed-pool Adaptive cohort continuation safely."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from datasets import load_from_disk  # type: ignore[import-untyped]

from graphfix.agent import OllamaConfig, OllamaTextGenerator
from graphfix.datasets import normalize_locbench_records
from graphfix.pipeline import (
    AdaptiveLocalizationResult,
    AdaptiveNavigationConfig,
    AdaptiveNavigationPipeline,
    AdvancedLocalizationPipeline,
    save_result,
)
from graphfix.repositories import RepositoryManager

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "raw" / "locbench_v1" / "dataset"
COHORT = ROOT / "results" / "localization_benchmark" / "evaluation_cohort.json"
CACHE = ROOT / "data" / "cache" / "repositories"
OUTPUT = (
    ROOT
    / "results"
    / "adaptive_navigation_experiments"
    / "fixed_pool_101_560"
)
START_INDEX = 100
END_INDEX = 560


def _load_valid_result(path: Path) -> AdaptiveLocalizationResult:
    return AdaptiveLocalizationResult.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _quarantine_invalid(path: Path) -> Path:
    invalid = OUTPUT / "recovery" / "invalid"
    invalid.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = invalid / f"{path.stem}.{stamp}.invalid.json"
    shutil.move(str(path), destination)
    return destination


def _save_atomically(result: AdaptiveLocalizationResult, destination: Path) -> None:
    partial = destination.with_suffix(".json.partial")
    save_result(result, partial)
    _load_valid_result(partial)
    partial.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-completions",
        type=int,
        help="Stop after this many new valid artifacts (useful for validation).",
    )
    args = parser.parse_args()
    if args.max_completions is not None and args.max_completions < 1:
        parser.error("--max-completions must be positive")

    cohort = json.loads(COHORT.read_text(encoding="utf-8"))["included_ids"]
    expected_ids: list[str] = cohort[START_INDEX:END_INDEX]
    output = OUTPUT / "adaptive"
    output.mkdir(parents=True, exist_ok=True)

    completed: set[str] = set()
    for instance_id in expected_ids:
        path = output / f"{instance_id}.json"
        if not path.exists():
            continue
        try:
            result = _load_valid_result(path)
            if result.instance_id != instance_id:
                raise ValueError(
                    f"filename expects {instance_id!r}, artifact has {result.instance_id!r}"
                )
            if result.adaptive_config != _adaptive_config():
                raise ValueError("artifact configuration differs from frozen configuration")
        except Exception as error:
            moved = _quarantine_invalid(path)
            print(f"INVALID {instance_id}: {error}; moved to {moved}", flush=True)
        else:
            completed.add(instance_id)

    raw = load_from_disk(str(SNAPSHOT))
    dataset = normalize_locbench_records(
        cast(Any, raw), raw_fields=cast(list[str], raw.column_names)
    )
    instances = {item.input.instance_id: item for item in dataset.instances}
    new_completions = 0
    failures = 0
    print(
        f"RESUME expected={len(expected_ids)} completed={len(completed)} "
        f"missing={len(expected_ids) - len(completed)}",
        flush=True,
    )

    for instance_id in expected_ids:
        if instance_id in completed:
            continue
        destination = output / f"{instance_id}.json"
        try:
            manager = RepositoryManager(CACHE)
            advanced = AdvancedLocalizationPipeline(manager)
            config = _adaptive_config()
            generator = OllamaTextGenerator(
                OllamaConfig(
                    model=config.model,
                    endpoint=config.endpoint,
                    timeout_seconds=config.timeout_seconds,
                )
            )
            pipeline = AdaptiveNavigationPipeline(
                manager,
                generator,
                config=config,
                hybrid_pipeline=advanced,
            )
            result = pipeline.localize(instances[instance_id].input)
            _save_atomically(result, destination)
        except Exception as error:
            failures += 1
            print(
                f"FAILED {instance_id}: {type(error).__name__}: {error}", flush=True
            )
        else:
            new_completions += 1
            completed.add(instance_id)
            print(
                f"COMPLETED {instance_id} total={len(completed)}/{len(expected_ids)}",
                flush=True,
            )
            if (
                args.max_completions is not None
                and new_completions >= args.max_completions
            ):
                break
        finally:
            if "pipeline" in locals():
                del pipeline
            if "advanced" in locals():
                del advanced
            if "manager" in locals():
                del manager
            if "generator" in locals():
                del generator
            gc.collect()

    print(
        f"STOP completed={len(completed)}/{len(expected_ids)} "
        f"new={new_completions} failures={failures}",
        flush=True,
    )
    return 0


def _adaptive_config() -> AdaptiveNavigationConfig:
    return AdaptiveNavigationConfig(
        provider="ollama",
        model="qwen2.5-coder:7b",
        endpoint="http://127.0.0.1:11434",
        temperature=0.0,
        seed=0,
        timeout_seconds=120.0,
        max_steps=10,
        max_retries=2,
        max_observed_nodes=8,
        max_snippet_chars=400,
        candidate_pool="fixed",
        max_expanded_candidates=5,
    )


if __name__ == "__main__":
    raise SystemExit(main())
