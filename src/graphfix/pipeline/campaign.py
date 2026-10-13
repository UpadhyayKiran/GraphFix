"""
Purpose:
    Runs resumable readiness and full benchmark campaigns.

Input:
    Dataset, repository manager, pipeline, and result path.

Output:
    Checkpoints, cohort, results, failures, and aggregates.

Pipeline stage:
    Full-benchmark reproducibility.
"""

from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from graphfix.datasets import LocBenchDataset
from graphfix.datasets.models import GraphFixInstance
from graphfix.evaluation import aggregate_metrics, extract_gold_locations
from graphfix.graph import build_repository_graph
from graphfix.parsing import PythonRepositoryParser
from graphfix.pipeline.localizer import BM25LocalizationPipeline, save_result
from graphfix.pipeline.models import EvaluationResult
from graphfix.repositories import RepositoryManager, RepositoryRequest

StageStatus = Literal["not_run", "success", "failed"]
OverallStatus = Literal["success", "failed"]


class CampaignModel(BaseModel):
    """Strict serialized base for long-running campaign artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ReadinessRecord(CampaignModel):
    """One checkpointed readiness outcome."""

    dataset_index: int = Field(ge=0)
    dataset_row: int = Field(ge=1)
    instance_id: str
    repository: str
    requested_base_commit: str
    resolved_head: str | None = None
    exact_sha_verified: bool = False
    checkout_status: StageStatus = "not_run"
    parser_status: StageStatus = "not_run"
    graph_status: StageStatus = "not_run"
    overall_status: OverallStatus
    failure_stage: str | None = None
    failure_category: str | None = None
    failure_message: str | None = None
    files_scanned: int = 0
    python_files_parsed: int = 0
    parse_failures: int = 0
    classes: int = 0
    functions: int = 0
    methods: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    elapsed_time: float = Field(ge=0.0)


class CohortExclusion(CampaignModel):
    """
    Purpose:
        Explains why one audited instance was not admitted to the frozen cohort.

    Contains:
        instance_id: The instance id value.
        reason: The reason value.

    Pipeline role:
        BM25 baseline and resumable campaign execution.
    """
    instance_id: str
    reason: str


class EvaluationCohort(CampaignModel):
    """
    Purpose:
        Freezes the exact benchmark IDs used by a localization evaluation run.

    Contains:
        benchmark: The benchmark value.
        split: The split value.
        created_at: The created at value.
        included_ids: The included ids value.
        excluded: The excluded value.
        environment: The environment value.

    Pipeline role:
        BM25 baseline and resumable campaign execution.
    """
    benchmark: str
    split: str
    created_at: datetime
    included_ids: tuple[str, ...]
    excluded: tuple[CohortExclusion, ...]
    environment: dict[str, str]


class EvaluationRunRecord(CampaignModel):
    """
    Purpose:
        Records the status, timing, and artifact path for one campaign instance.

    Contains:
        dataset_index: The dataset index value.
        instance_id: The instance id value.
        status: The status value.
        elapsed_time: The elapsed time value.
        failure_category: The failure category value.
        failure_message: The failure message value.

    Pipeline role:
        BM25 baseline and resumable campaign execution.
    """
    dataset_index: int = Field(ge=0)
    instance_id: str
    status: OverallStatus
    elapsed_time: float = Field(ge=0.0)
    failure_category: str | None = None
    failure_message: str | None = None


def _now() -> datetime:
    """
    Purpose:
        Return the current UTC time in a stable text format.

    Returns:
        A `datetime` result produced by the operation.

    Pipeline role:
        BM25 baseline and resumable campaign execution.
    """
    return datetime.now(UTC)


def _environment() -> dict[str, str]:
    """
    Purpose:
        Collect the small environment summary stored with campaign artifacts.

    Returns:
        A `dict[str, str]` result produced by the operation.

    Pipeline role:
        BM25 baseline and resumable campaign execution.
    """
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def _classify_failure(error: BaseException | str) -> str:
    """
    Purpose:
        Map a failure message to a short campaign failure category.

    Args:
        error: The error value used by this operation.

    Returns:
        A `str` result produced by the operation.

    Pipeline role:
        BM25 baseline and resumable campaign execution.
    """
    message = str(error).casefold()
    if "filename too long" in message or "unable to create file" in message:
        return "windows_path_length"
    if isinstance(error, RecursionError) or "recursion depth" in message:
        return "recursion_depth"
    if "timed out" in message or "timeout" in message:
        return "timeout"
    if "commit" in message and ("not found" in message or "unavailable" in message):
        return "missing_commit"
    return "unknown"


def _failure_record(
    index: int,
    instance: GraphFixInstance,
    started: float,
    *,
    stage: str,
    category: str,
    message: str,
    resolved_head: str | None = None,
    checkout_status: StageStatus = "not_run",
    parser_status: StageStatus = "not_run",
) -> ReadinessRecord:
    """
    Purpose:
        Build a serializable record for one failed campaign instance.

    Args:
        index: The index value used by this operation.
        instance: One validated GraphFix benchmark instance.
        started: The started value used by this operation.
        stage: The stage value used by this operation.
        category: The category value used by this operation.
        message: The message value used by this operation.
        resolved_head: The resolved head value used by this operation.
        checkout_status: The checkout status value used by this operation.
        parser_status: The parser status value used by this operation.

    Returns:
        A `ReadinessRecord` result produced by the operation.

    Pipeline role:
        BM25 baseline and resumable campaign execution.
    """
    return ReadinessRecord(
        dataset_index=index,
        dataset_row=index + 1,
        instance_id=instance.input.instance_id,
        repository=instance.input.repo,
        requested_base_commit=instance.input.base_commit,
        resolved_head=resolved_head,
        exact_sha_verified=bool(
            resolved_head
            and resolved_head.casefold() == instance.input.base_commit.casefold()
        ),
        checkout_status=checkout_status,
        parser_status=parser_status,
        graph_status="failed" if stage == "graph" else "not_run",
        overall_status="failed",
        failure_stage=stage,
        failure_category=category,
        failure_message=message[:4000],
        elapsed_time=perf_counter() - started,
    )


def audit_instance(
    index: int,
    instance: GraphFixInstance,
    repository_manager: RepositoryManager,
    parser: PythonRepositoryParser,
) -> ReadinessRecord:
    """Audit reconstruction, parsing, and graph construction without retrieval."""
    started = perf_counter()
    checkout = repository_manager.reconstruct(
        RepositoryRequest.from_graphfix_input(instance.input)
    )
    if not checkout.success or not checkout.checkout_path or not checkout.resolved_commit:
        failure = checkout.failure
        message = failure.message if failure else "unknown checkout failure"
        category = failure.category.casefold() if failure else _classify_failure(message)
        if _classify_failure(message) != "unknown":
            category = _classify_failure(message)
        return _failure_record(
            index,
            instance,
            started,
            stage="checkout",
            category=category,
            message=message,
            resolved_head=checkout.resolved_commit,
            checkout_status="failed",
        )
    try:
        parsed = parser.parse(Path(checkout.checkout_path))
    except Exception as error:
        return _failure_record(
            index,
            instance,
            started,
            stage="parser",
            category=(
                _classify_failure(error)
                if _classify_failure(error) != "unknown"
                else "parser_failure"
            ),
            message=f"{type(error).__name__}: {error}",
            resolved_head=checkout.resolved_commit,
            checkout_status="success",
            parser_status="failed",
        )
    try:
        graph = build_repository_graph(parsed, repository_name=instance.input.repo)
    except Exception as error:
        record = _failure_record(
            index,
            instance,
            started,
            stage="graph",
            category=(
                _classify_failure(error)
                if _classify_failure(error) != "unknown"
                else "graph_failure"
            ),
            message=f"{type(error).__name__}: {error}",
            resolved_head=checkout.resolved_commit,
            checkout_status="success",
            parser_status="success",
        )
        return record.model_copy(
            update={
                "files_scanned": parsed.stats.files_scanned,
                "python_files_parsed": parsed.stats.python_files_parsed,
                "parse_failures": parsed.stats.parse_failures,
                "classes": parsed.stats.classes,
                "functions": parsed.stats.functions,
                "methods": parsed.stats.methods,
            }
        )
    return ReadinessRecord(
        dataset_index=index,
        dataset_row=index + 1,
        instance_id=instance.input.instance_id,
        repository=instance.input.repo,
        requested_base_commit=instance.input.base_commit,
        resolved_head=checkout.resolved_commit,
        exact_sha_verified=(
            checkout.resolved_commit.casefold() == instance.input.base_commit.casefold()
        ),
        checkout_status="success",
        parser_status="success",
        graph_status="success",
        overall_status="success",
        files_scanned=parsed.stats.files_scanned,
        python_files_parsed=parsed.stats.python_files_parsed,
        parse_failures=parsed.stats.parse_failures,
        classes=parsed.stats.classes,
        functions=parsed.stats.functions,
        methods=parsed.stats.methods,
        graph_nodes=graph.stats.total_nodes,
        graph_edges=graph.stats.total_edges,
        elapsed_time=perf_counter() - started,
    )


def _append_model(path: Path, model: CampaignModel) -> None:
    """
    Purpose:
        Append one Pydantic model as a JSON line, creating parent folders when needed.

    Args:
        path: The file or directory path to use.
        model: Pydantic model to serialize.

    Returns:
        Nothing. The operation updates state, writes output, or performs a check directly.

    Pipeline role:
        BM25 baseline and resumable campaign execution.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(model.model_dump_json() + "\n")
        handle.flush()


def _latest_records(path: Path, model: type[ReadinessRecord]) -> dict[str, ReadinessRecord]:
    """
    Purpose:
        Keep only the latest checkpoint record for each instance ID.

    Args:
        path: The file or directory path to use.
        model: Pydantic model to serialize.

    Returns:
        A `dict[str, ReadinessRecord]` result produced by the operation.

    Pipeline role:
        BM25 baseline and resumable campaign execution.
    """
    records: dict[str, ReadinessRecord] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = model.model_validate_json(line)
            records[record.instance_id] = record
    return records


def _write_json(path: Path, value: Any) -> None:
    """
    Purpose:
        Write formatted UTF-8 JSON and create its parent folder.

    Args:
        path: The file or directory path to use.
        value: The value to check or convert.

    Returns:
        Nothing. The operation updates state, writes output, or performs a check directly.

    Pipeline role:
        BM25 baseline and resumable campaign execution.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def export_readiness(records: Sequence[ReadinessRecord], results_dir: Path) -> None:
    """Export deterministic JSON and CSV views of the latest audit checkpoints."""
    ordered = sorted(records, key=lambda item: item.dataset_index)
    payload = [item.model_dump(mode="json") for item in ordered]
    _write_json(results_dir / "readiness_audit.json", payload)
    csv_path = results_dir / "readiness_audit.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ReadinessRecord.model_fields))
        writer.writeheader()
        writer.writerows(payload)
    temporary.replace(csv_path)


def run_readiness_audit(
    dataset: LocBenchDataset,
    repository_manager: RepositoryManager,
    results_dir: Path,
    *,
    retry_failures: bool = False,
) -> tuple[ReadinessRecord, ...]:
    """Run or resume the readiness audit, checkpointing every attempted row."""
    checkpoint = results_dir / "readiness_checkpoints.jsonl"
    latest = _latest_records(checkpoint, ReadinessRecord)
    parser = PythonRepositoryParser()
    total = len(dataset.instances)
    for index, instance in enumerate(dataset.instances):
        existing = latest.get(instance.input.instance_id)
        if existing and (existing.overall_status == "success" or not retry_failures):
            print(f"[{index + 1}/{total}] {instance.input.instance_id}: resume-skip")
            continue
        print(f"[{index + 1}/{total}] {instance.input.instance_id}: auditing", flush=True)
        record = audit_instance(index, instance, repository_manager, parser)
        _append_model(checkpoint, record)
        latest[record.instance_id] = record
        print(
            f"[{index + 1}/{total}] {record.instance_id}: {record.overall_status} "
            f"({record.elapsed_time:.2f}s)",
            flush=True,
        )
    records = tuple(
        latest[item.input.instance_id]
        for item in dataset.instances
        if item.input.instance_id in latest
    )
    export_readiness(records, results_dir)
    return records


def freeze_cohort(
    dataset: LocBenchDataset,
    records: Sequence[ReadinessRecord],
    results_dir: Path,
) -> EvaluationCohort:
    """Freeze all infrastructure-ready rows with usable evaluation gold."""
    by_id = {record.instance_id: record for record in records}
    included: list[str] = []
    excluded: list[CohortExclusion] = []
    for instance in dataset.instances:
        instance_id = instance.input.instance_id
        record = by_id.get(instance_id)
        if not record or record.overall_status != "success":
            reason = record.failure_category if record else "missing_audit_record"
            excluded.append(CohortExclusion(instance_id=instance_id, reason=reason or "unknown"))
            continue
        gold = extract_gold_locations(instance.gold)
        if not gold.files and not gold.functions:
            excluded.append(CohortExclusion(instance_id=instance_id, reason="unusable_gold"))
            continue
        included.append(instance_id)
    cohort = EvaluationCohort(
        benchmark=dataset.dataset_name,
        split=dataset.split,
        created_at=_now(),
        included_ids=tuple(included),
        excluded=tuple(excluded),
        environment=_environment(),
    )
    _write_json(results_dir / "evaluation_cohort.json", cohort.model_dump(mode="json"))
    return cohort


def _load_run_records(path: Path) -> dict[str, EvaluationRunRecord]:
    """
    Purpose:
        Read completed campaign run records so evaluation can resume safely.

    Args:
        path: The file or directory path to use.

    Returns:
        A `dict[str, EvaluationRunRecord]` result produced by the operation.

    Pipeline role:
        BM25 baseline and resumable campaign execution.
    """
    records: dict[str, EvaluationRunRecord] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = EvaluationRunRecord.model_validate_json(line)
                records[record.instance_id] = record
    return records


def _load_evaluation(
    path: Path, expected_config: Any | None = None,
) -> EvaluationResult | None:
    """Load a valid checkpoint, optionally requiring the active retrieval configuration."""
    if not path.exists():
        return None
    try:
        result = EvaluationResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return None
    if expected_config is not None and result.localization.retrieval_config != expected_config:
        return None
    return result


def run_full_evaluation(
    dataset: LocBenchDataset,
    cohort: EvaluationCohort,
    pipeline: BM25LocalizationPipeline,
    results_dir: Path,
    *,
    retry_failures: bool = False,
) -> tuple[EvaluationResult, ...]:
    """Run or resume fixed-BM25 evaluation and persist every completed result."""
    by_id = {item.input.instance_id: (index, item) for index, item in enumerate(dataset.instances)}
    run_path = results_dir / "run_records.jsonl"
    run_records = _load_run_records(run_path)
    instance_dir = results_dir / "localization_results"
    total = len(cohort.included_ids)
    for position, instance_id in enumerate(cohort.included_ids, start=1):
        index, instance = by_id[instance_id]
        output = instance_dir / f"{instance_id}.json"
        completed = _load_evaluation(output, pipeline.config)
        prior = run_records.get(instance_id)
        if completed is not None:
            print(f"[{position}/{total}] {instance_id}: resume-skip", flush=True)
            continue
        if prior and prior.status == "failed" and not retry_failures:
            print(f"[{position}/{total}] {instance_id}: failure-skip", flush=True)
            continue
        print(f"[{position}/{total}] {instance_id}: evaluating", flush=True)
        started = perf_counter()
        try:
            result = pipeline.evaluate(instance)
        except Exception as error:
            record = EvaluationRunRecord(
                dataset_index=index,
                instance_id=instance_id,
                status="failed",
                elapsed_time=perf_counter() - started,
                failure_category=_classify_failure(error),
                failure_message=f"{type(error).__name__}: {error}"[:4000],
            )
        else:
            save_result(result, output)
            record = EvaluationRunRecord(
                dataset_index=index,
                instance_id=instance_id,
                status="success",
                elapsed_time=perf_counter() - started,
            )
        _append_model(run_path, record)
        run_records[instance_id] = record
        print(
            f"[{position}/{total}] {instance_id}: {record.status} "
            f"({record.elapsed_time:.2f}s)",
            flush=True,
        )
    collected: list[EvaluationResult] = []
    for instance_id in cohort.included_ids:
        loaded = _load_evaluation(
            instance_dir / f"{instance_id}.json", pipeline.config,
        )
        if loaded is not None:
            collected.append(loaded)
    evaluations = tuple(collected)
    _export_evaluation(dataset, cohort, evaluations, run_records, results_dir)
    return evaluations


def _sum_stats(evaluations: Sequence[EvaluationResult], section: str, field: str) -> int:
    """
    Purpose:
        Add numeric parser or graph statistics across successful instances.

    Args:
        evaluations: The evaluations value used by this operation.
        section: The section value used by this operation.
        field: Name of the dataset field being processed.

    Returns:
        A `int` result produced by the operation.

    Pipeline role:
        BM25 baseline and resumable campaign execution.
    """
    return sum(
        int(getattr(getattr(item.localization, section), field)) for item in evaluations
    )


def _export_evaluation(
    dataset: LocBenchDataset,
    cohort: EvaluationCohort,
    evaluations: Sequence[EvaluationResult],
    run_records: dict[str, EvaluationRunRecord],
    results_dir: Path,
) -> None:
    """
    Purpose:
        Export one evaluation to per-instance files, a localization stream, and run records.

    Args:
        dataset: The normalized Loc-Bench dataset.
        cohort: The cohort value used by this operation.
        evaluations: The evaluations value used by this operation.
        run_records: The run records value used by this operation.
        results_dir: The results dir value used by this operation.

    Returns:
        Nothing. The operation updates state, writes output, or performs a check directly.

    Pipeline role:
        BM25 baseline and resumable campaign execution.
    """
    metrics = aggregate_metrics([item.metrics for item in evaluations])
    durations = [
        run_records[item.localization.instance_id].elapsed_time
        for item in evaluations
        if item.localization.instance_id in run_records
    ]
    parser_totals = {
        field: _sum_stats(evaluations, "parser_stats", field)
        for field in (
            "files_scanned",
            "python_files_parsed",
            "parse_failures",
            "classes",
            "functions",
            "methods",
            "imports",
            "calls",
        )
    }
    graph_totals = {
        field: _sum_stats(evaluations, "graph_stats", field)
        for field in ("total_nodes", "total_edges")
    }
    summary = {
        "benchmark": dataset.dataset_name,
        "split": dataset.split,
        "cohort_size": len(cohort.included_ids),
        "completed": len(evaluations),
        "runtime_failures": sum(
            record.status == "failed"
            for instance_id, record in run_records.items()
            if instance_id in cohort.included_ids
        ),
        "metrics": metrics.model_dump(mode="json"),
        "runtime": {
            "total_seconds": sum(durations),
            "mean_seconds": statistics.fmean(durations) if durations else 0.0,
            "median_seconds": statistics.median(durations) if durations else 0.0,
        },
        "parser_totals": parser_totals,
        "parser_averages": {
            key: value / len(evaluations) if evaluations else 0.0
            for key, value in parser_totals.items()
        },
        "graph_totals": graph_totals,
        "graph_averages": {
            key: value / len(evaluations) if evaluations else 0.0
            for key, value in graph_totals.items()
        },
        "generated_at": _now().isoformat(),
        "environment": _environment(),
    }
    _write_json(results_dir / "metrics.json", summary)
    localization_results = results_dir / "localization_results.jsonl"
    localization_results.write_text(
        "".join(item.model_dump_json() + "\n" for item in evaluations),
        encoding="utf-8",
    )
    failures = [
        record.model_dump(mode="json")
        for record in run_records.values()
        if record.status == "failed"
    ]
    (results_dir / "failures.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    failure_counts = Counter(item["failure_category"] for item in failures)
    _write_json(
        results_dir / "run_metadata.json",
        {
            "generated_at": _now().isoformat(),
            "environment": _environment(),
            "failure_counts": dict(sorted(failure_counts.items(), key=lambda item: str(item[0]))),
        },
    )
    runtime = summary["runtime"]
    assert isinstance(runtime, dict)
    total_seconds = float(runtime["total_seconds"])
    mean_seconds = float(runtime["mean_seconds"])
    median_seconds = float(runtime["median_seconds"])
    markdown = (
        "# Full BM25 Baseline Summary\n\n"
        f"- Cohort: {len(cohort.included_ids)}\n"
        f"- Completed: {len(evaluations)}\n"
        f"- Runtime failures: {summary['runtime_failures']}\n"
        f"- Total runtime: {total_seconds:.3f} seconds\n"
        f"- Mean runtime: {mean_seconds:.3f} seconds\n"
        f"- Median runtime: {median_seconds:.3f} seconds\n\n"
        "Metrics are stored in `bm25_full_metrics.json`.\n"
    )
    (results_dir / "bm25_full_summary.md").write_text(markdown, encoding="utf-8")


def readiness_counts(records: Iterable[ReadinessRecord]) -> dict[str, int]:
    """Summarize readiness outcomes for progress reporting."""
    materialized = tuple(records)
    return {
        "records": len(materialized),
        "checkout_success": sum(item.checkout_status == "success" for item in materialized),
        "exact_sha_verified": sum(item.exact_sha_verified for item in materialized),
        "parser_success": sum(item.parser_status == "success" for item in materialized),
        "graph_success": sum(item.graph_status == "success" for item in materialized),
        "fully_evaluable": sum(item.overall_status == "success" for item in materialized),
        "failed": sum(item.overall_status == "failed" for item in materialized),
    }
