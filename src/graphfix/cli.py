"""
Purpose:
    Provides the canonical command-line interface for GraphFix demonstrations.

Input:
    A local Loc-Bench snapshot, instance IDs, paths, and BM25 settings.

Output:
    Stage summaries, rankings, evaluations, and JSON result files.

Pipeline stage:
    User-facing entry point across the localization pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, cast

from datasets import load_from_disk  # type: ignore[import-untyped]

from graphfix.agent import OllamaConfig, OllamaTextGenerator
from graphfix.datasets import LocBenchDataset, normalize_locbench_records
from graphfix.evaluation import aggregate_metrics, extract_gold_locations
from graphfix.graph import build_repository_graph
from graphfix.parsing import PythonRepositoryParser
from graphfix.pipeline import (
    AdaptiveLocalizationResult,
    AdaptiveNavigationConfig,
    AdaptiveNavigationPipeline,
    AdvancedLocalizationPipeline,
    AdvancedLocalizationResult,
    AdvancedRetrievalConfig,
    BM25LocalizationPipeline,
    EvaluationResult,
    LocalizationResult,
    RetrievalConfig,
    save_result,
)
from graphfix.repositories import RepositoryManager, RepositoryRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = PROJECT_ROOT / "data" / "raw" / "locbench_v1" / "dataset"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "cache" / "repositories"
DEFAULT_BM25_RESULTS = PROJECT_ROOT / "results" / "bm25_baseline"
DEFAULT_COMPARISON_RESULTS = PROJECT_ROOT / "results" / "retrieval_comparison"
DEFAULT_ADAPTIVE_RESULTS = PROJECT_ROOT / "results" / "adaptive_navigation"

LOGGER = logging.getLogger("graphfix")


def _load_local_dataset(snapshot: Path, *, limit: int | None = None) -> LocBenchDataset:
    """
    Purpose:
        Load and normalize the local Loc-Bench snapshot for a CLI command.

    Args:
        snapshot: The snapshot value used by this operation.
        limit: The limit value used by this operation.

    Returns:
        A `LocBenchDataset` result produced by the operation.

    Pipeline role:
        User-facing CLI execution across the localization stages.
    """
    raw = load_from_disk(str(snapshot))
    return normalize_locbench_records(
        cast(Any, raw),
        raw_fields=cast(list[str], raw.column_names),
        limit=limit,
    )


def _find_instance(dataset: LocBenchDataset, instance_id: str) -> Any:
    """
    Purpose:
        Find one benchmark instance by its unique ID.

    Args:
        dataset: The normalized Loc-Bench dataset.
        instance_id: The unique Loc-Bench instance ID.

    Returns:
        A `Any` result produced by the operation.

    Pipeline role:
        User-facing CLI execution across the localization stages.
    """
    return next(
        (instance for instance in dataset.instances if instance.input.instance_id == instance_id),
        None,
    )


def _require_instance(args: argparse.Namespace) -> Any:
    """Load one requested row or stop with a clear unknown-ID message."""
    instance = _find_instance(_load_local_dataset(args.snapshot), args.instance_id)
    if instance is None:
        raise SystemExit(f"Unknown instance ID: {args.instance_id}")
    return instance


def _reconstruct(args: argparse.Namespace, instance: Any) -> Any:
    """Reconstruct one exact commit and reject a failed or mismatched checkout."""
    checkout = RepositoryManager(args.cache_root).reconstruct(
        RepositoryRequest.from_graphfix_input(instance.input)
    )
    if not checkout.success or not checkout.checkout_path or not checkout.resolved_commit:
        message = checkout.failure.message if checkout.failure else "unknown reconstruction failure"
        raise SystemExit(message)
    if checkout.resolved_commit.casefold() != instance.input.base_commit.casefold():
        raise SystemExit("HEAD does not equal the requested base_commit")
    return checkout


def _pipeline(args: argparse.Namespace) -> BM25LocalizationPipeline:
    """
    Purpose:
        Build the BM25 localization pipeline with the command-line settings.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A `BM25LocalizationPipeline` result produced by the operation.

    Pipeline role:
        User-facing CLI execution across the localization stages.
    """
    config = RetrievalConfig(
        top_k=args.top_k,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
    )
    return BM25LocalizationPipeline(RepositoryManager(args.cache_root), config=config)


def _print_rankings(title: str, rankings: tuple[Any, ...]) -> None:
    """
    Purpose:
        Print a readable numbered list of BM25 candidates.

    Args:
        title: The title value used by this operation.
        rankings: The rankings value used by this operation.

    Returns:
        Nothing. The operation updates state, writes output, or performs a check directly.

    Pipeline role:
        User-facing CLI execution across the localization stages.
    """
    print(title)
    if not rankings:
        print("  (none)")
    for item in rankings:
        symbol = f" :: {item.qualified_name}" if item.qualified_name and item.symbol_name else ""
        lines = (
            f" [{item.start_line}-{item.end_line}]" if item.start_line is not None else ""
        )
        print(f"  {item.rank}. {item.file_path}{symbol}{lines} score={item.score:.4f}")


def _print_evaluation(result: EvaluationResult) -> None:
    """
    Purpose:
        Print the complete single-instance localization output.

    Args:
        result: Pipeline result to save or summarize.

    Returns:
        Nothing. The operation updates state, writes output, or performs a check directly.

    Pipeline role:
        User-facing CLI execution across the localization stages.
    """
    localization = result.localization
    print("=" * 72)
    print("GRAPHFIX SOFTWARE ISSUE LOCALIZATION")
    print("=" * 72)
    print("INSTANCE")
    print(f"  ID: {localization.instance_id}")
    print(f"  Repository: {localization.repository}")
    print(f"  base_commit: {localization.base_commit}")
    print("ISSUE")
    print(f"  {localization.issue}")
    print("REPOSITORY")
    print(f"  checked-out HEAD: {localization.checked_out_commit}")
    verified = localization.checked_out_commit.casefold() == localization.base_commit.casefold()
    print(f"  verified equals base_commit: {verified}")
    parser = localization.parser_stats
    print("PARSER")
    print(
        f"  scanned={parser.files_scanned} parsed={parser.python_files_parsed} "
        f"failures={parser.parse_failures} classes={parser.classes} "
        f"functions={parser.functions} methods={parser.methods}"
    )
    graph = localization.graph_stats
    print("GRAPH")
    print(f"  nodes={graph.total_nodes} {graph.nodes_by_type}")
    print(f"  edges={graph.total_edges} {graph.edges_by_type}")
    _print_rankings("BM25 FILE TOP-K", localization.file_rankings)
    _print_rankings("BM25 FUNCTION TOP-K", localization.function_rankings)
    print("GOLD - EVALUATION ONLY")
    print(f"  files: {list(result.gold.files)}")
    print(f"  functions: {list(result.gold.functions)}")
    print("RESULT")
    print(f"  file: {result.metrics.file.model_dump()}")
    print(f"  function: {result.metrics.function.model_dump()}")


def _run_inspect(args: argparse.Namespace) -> int:
    """
    Purpose:
        Handle `inspect-instance` by printing safe input and a separate gold summary.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A `int` result produced by the operation.

    Pipeline role:
        User-facing CLI execution across the localization stages.
    """
    instance = _require_instance(args)
    gold = extract_gold_locations(instance.gold)
    payload = {"input": instance.input.model_dump(), "gold_summary": gold.model_dump()}
    print(json.dumps(payload, indent=2))
    return 0


def _run_reconstruct(args: argparse.Namespace) -> int:
    """Show the exact requested and resolved commits for one instance."""
    instance = _require_instance(args)
    checkout = _reconstruct(args, instance)
    print(json.dumps(checkout.model_dump(mode="json"), indent=2))
    return 0


def _run_parse(args: argparse.Namespace) -> int:
    """Reconstruct and print deterministic parser statistics."""
    instance = _require_instance(args)
    checkout = _reconstruct(args, instance)
    parsed = PythonRepositoryParser().parse(Path(checkout.checkout_path))
    print(json.dumps({
        "instance_id": instance.input.instance_id,
        "head": checkout.resolved_commit,
        "statistics": parsed.stats.model_dump(mode="json"),
        "failures": list(parsed.failures),
    }, indent=2))
    return 0


def _run_graph(args: argparse.Namespace) -> int:
    """Reconstruct, parse, and print heterogeneous graph statistics."""
    instance = _require_instance(args)
    checkout = _reconstruct(args, instance)
    parsed = PythonRepositoryParser().parse(Path(checkout.checkout_path))
    graph = build_repository_graph(parsed, repository_name=instance.input.repo)
    print(json.dumps({
        "instance_id": instance.input.instance_id,
        "head": checkout.resolved_commit,
        "graph": graph.stats.model_dump(mode="json"),
        "bm25_uses_graph": False,
    }, indent=2))
    return 0


def _run_localize(args: argparse.Namespace) -> int:
    """
    Purpose:
        Run the selected gold-free localization method and save its JSON result.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A `int` result produced by the operation.

    Raises:
        SystemExit: If the input or operation cannot be completed safely.

    Pipeline role:
        User-facing CLI execution across the localization stages.
    """
    dataset = _load_local_dataset(args.snapshot)
    instance = _find_instance(dataset, args.instance_id)
    if instance is None:
        raise SystemExit(f"Unknown instance ID: {args.instance_id}")
    result: LocalizationResult | AdvancedLocalizationResult | AdaptiveLocalizationResult
    if args.method == "bm25":
        result = _pipeline(args).localize(instance.input)
        results_dir = args.results_dir
    elif args.method == "adaptive":
        if not args.ollama_model:
            raise SystemExit("--ollama-model is required for adaptive localization")
        ollama = OllamaConfig(
            model=args.ollama_model,
            endpoint=args.ollama_endpoint,
            timeout_seconds=args.controller_timeout,
        )
        result = AdaptiveNavigationPipeline(
            RepositoryManager(args.cache_root),
            OllamaTextGenerator(ollama),
            config=AdaptiveNavigationConfig(
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
            ),
        ).localize(instance.input)
        results_dir = (
            DEFAULT_ADAPTIVE_RESULTS
            if args.results_dir == DEFAULT_BM25_RESULTS
            else args.results_dir
        )
    else:
        advanced_config = AdvancedRetrievalConfig(
            top_k=args.top_k,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
        )
        result = AdvancedLocalizationPipeline(
            RepositoryManager(args.cache_root), config=advanced_config
        ).localize(instance.input, method=args.method)
        results_dir = (
            DEFAULT_COMPARISON_RESULTS
            if args.results_dir == DEFAULT_BM25_RESULTS
            else args.results_dir
        )
    path = results_dir / f"{result.instance_id}.{args.method}.localization.json"
    save_result(result, path)
    method_label = args.method.upper()
    _print_rankings(f"{method_label} FILE TOP-K", result.file_rankings)
    _print_rankings(f"{method_label} FUNCTION TOP-K", result.function_rankings)
    print("LOCALIZATION COMPLETE")
    print(f"Result: {path}")
    return 0


def _run_demo(args: argparse.Namespace) -> int:
    """
    Purpose:
        Handle `demo`, evaluate one instance, print it, and save its JSON result.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A `int` result produced by the operation.

    Raises:
        SystemExit: If the input or operation cannot be completed safely.

    Pipeline role:
        User-facing CLI execution across the localization stages.
    """
    dataset = _load_local_dataset(args.snapshot)
    instance = _find_instance(dataset, args.instance_id)
    if instance is None:
        raise SystemExit(f"Unknown instance ID: {args.instance_id}")
    result = _pipeline(args).evaluate(instance)
    path = args.results_dir / f"{result.localization.instance_id}.evaluation.json"
    save_result(result, path)
    _print_evaluation(result)
    print(f"ARTIFACT: {path}")
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    """
    Purpose:
        Handle `evaluate` for a bounded set of instances. Save per-instance and aggregate results.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A `int` result produced by the operation.

    Raises:
        SystemExit: If the input or operation cannot be completed safely.

    Pipeline role:
        User-facing CLI execution across the localization stages.
    """
    dataset = _load_local_dataset(args.snapshot)
    selected = list(dataset.instances)
    if args.instance_id:
        wanted = set(args.instance_id)
        selected = [item for item in selected if item.input.instance_id in wanted]
        missing = wanted.difference(item.input.instance_id for item in selected)
        if missing:
            raise SystemExit(f"Unknown instance IDs: {', '.join(sorted(missing))}")
    selected = selected[: args.limit]
    pipeline = _pipeline(args)
    evaluations: list[EvaluationResult] = []
    for index, instance in enumerate(selected, start=1):
        LOGGER.info("Evaluating %s (%d/%d)", instance.input.instance_id, index, len(selected))
        try:
            result = pipeline.evaluate(instance)
        except RuntimeError as error:
            LOGGER.error("%s", error)
            continue
        evaluations.append(result)
        save_result(
            result,
            args.results_dir / "instances" / f"{result.localization.instance_id}.json",
        )
    aggregate = aggregate_metrics([item.metrics for item in evaluations])
    summary = {
        "requested": len(selected),
        "completed": len(evaluations),
        "configuration": pipeline.config.model_dump(),
        "metrics": aggregate.model_dump(),
        "instance_ids": [item.localization.instance_id for item in evaluations],
    }
    summary_path = args.results_dir / "evaluation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    Purpose:
        Define the GraphFix commands, options, defaults, and command handlers.

    Returns:
        A `argparse.ArgumentParser` result produced by the operation.

    Pipeline role:
        User-facing CLI execution across the localization stages.
    """
    parser = argparse.ArgumentParser(prog="graphfix", description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_BM25_RESULTS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect-instance", help="Show normalized input and gold summary")
    inspect.add_argument("instance_id")
    inspect.set_defaults(handler=_run_inspect)

    reconstruct = commands.add_parser("reconstruct", help="Verify one exact historical checkout")
    reconstruct.add_argument("instance_id")
    reconstruct.set_defaults(handler=_run_reconstruct)

    parse = commands.add_parser("parse", help="Parse one verified historical checkout")
    parse.add_argument("instance_id")
    parse.set_defaults(handler=_run_parse)

    graph = commands.add_parser("graph", help="Build and summarize one repository graph")
    graph.add_argument("instance_id")
    graph.set_defaults(handler=_run_graph)

    localize = commands.add_parser("localize", help="Run a gold-free localization method")
    localize.add_argument("instance_id")
    localize.add_argument(
        "--method",
        choices=("bm25", "dense", "hybrid", "graph", "adaptive"),
        default="bm25",
    )
    localize.add_argument("--ollama-model")
    localize.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    localize.add_argument("--controller-timeout", type=float, default=120.0)
    localize.add_argument("--max-steps", type=int, default=10)
    localize.add_argument("--max-retries", type=int, default=2)
    localize.add_argument(
        "--candidate-pool", choices=("fixed", "expanded"), default="fixed"
    )
    localize.add_argument("--max-expanded-candidates", type=int, default=5)
    localize.set_defaults(handler=_run_localize)

    demo = commands.add_parser("demo", help="Run one readable localization and evaluation")
    demo.add_argument("instance_id")
    demo.set_defaults(handler=_run_demo)

    evaluate = commands.add_parser("evaluate", help="Evaluate a bounded instance subset")
    evaluate.add_argument("--limit", type=int, default=3)
    evaluate.add_argument("--instance-id", action="append")
    evaluate.set_defaults(handler=_run_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Purpose:
        Parse command-line arguments, validate numeric options, and run the selected handler.

    Args:
        argv: Optional command-line arguments; uses the process arguments when omitted.

    Returns:
        A `int` result produced by the operation.

    Pipeline role:
        User-facing CLI execution across the localization stages.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if getattr(args, "limit", 1) <= 0:
        parser.error("--limit must be positive")
    if getattr(args, "max_steps", 1) <= 0:
        parser.error("--max-steps must be positive")
    if getattr(args, "max_retries", 0) < 0:
        parser.error("--max-retries must be non-negative")
    if getattr(args, "max_expanded_candidates", 1) <= 0:
        parser.error("--max-expanded-candidates must be positive")
    if getattr(args, "controller_timeout", 1.0) <= 0:
        parser.error("--controller-timeout must be positive")
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )
    return int(args.handler(args))
