"""
Purpose:
    Calculates per-instance and aggregate ranking metrics.

Input:
    Completed rankings and normalized gold.

Output:
    Hit@1, Recall@5, Recall@10, and MRR.

Pipeline stage:
    Post-localization evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from graphfix.evaluation.matching import GoldLocations, normalize_repository_path
from graphfix.retrieval import RankedCandidate


class MetricModel(BaseModel):
    """
    Purpose:
        Base model that rejects unexpected metric fields.

    Pipeline role:
        Post-localization comparison with gold locations.
    """
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class LevelMetrics(MetricModel):
    """
    Purpose:
        Stores ranking metrics for either files or functions.

    Contains:
        hit_at_1: The hit at 1 value.
        recall_at_5: The recall at 5 value.
        recall_at_10: The recall at 10 value.
        mrr: The mrr value.
        gold_count: The gold count value.
        first_relevant_rank: The first relevant rank value.

    Pipeline role:
        Post-localization comparison with gold locations.
    """
    hit_at_1: float = Field(ge=0.0, le=1.0)
    recall_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_10: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    gold_count: int = Field(ge=0)
    first_relevant_rank: int | None = Field(default=None, ge=1)


class LocalizationMetrics(MetricModel):
    """
    Purpose:
        Groups file-level and function-level metrics for one instance.

    Contains:
        file: The file value.
        function: The function value.

    Pipeline role:
        Post-localization comparison with gold locations.
    """
    file: LevelMetrics
    function: LevelMetrics


class AggregateMetrics(MetricModel):
    """
    Purpose:
        Stores macro-averaged metrics and the number of evaluated instances.

    Contains:
        instances: The instances value.
        file_hit_at_1: The file hit at 1 value.
        file_recall_at_5: The file recall at 5 value.
        file_recall_at_10: The file recall at 10 value.
        file_mrr: The file mrr value.
        function_hit_at_1: The function hit at 1 value.
        function_recall_at_5: The function recall at 5 value.
        function_recall_at_10: The function recall at 10 value.
        function_mrr: The function mrr value.

    Pipeline role:
        Post-localization comparison with gold locations.
    """
    instances: int = Field(ge=0)
    file_hit_at_1: float
    file_recall_at_5: float
    file_recall_at_10: float
    file_mrr: float
    function_hit_at_1: float
    function_recall_at_5: float
    function_recall_at_10: float
    function_mrr: float


def _level_metrics(predicted: Sequence[str], gold: set[str]) -> LevelMetrics:
    """
    Purpose:
        Calculate Hit@1, Recall@5, Recall@10, and MRR for one ranking level.

    Args:
        predicted: Predicted locations to compare with gold.
        gold: Gold locations used only during evaluation.

    Returns:
        A `LevelMetrics` result produced by the operation.

    Pipeline role:
        Post-localization comparison with gold locations.
    """
    if not gold:
        return LevelMetrics(
            hit_at_1=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            mrr=0.0,
            gold_count=0,
        )
    relevant_ranks = [index for index, value in enumerate(predicted, start=1) if value in gold]
    first = min(relevant_ranks) if relevant_ranks else None

    def recall(k: int) -> float:
        """
        Purpose:
        Return the fraction of unique gold locations found in the first k ranked candidates.

        Args:
            k: Ranking cutoff used by the metric.

        Returns:
            A `float` result produced by the operation.

        Pipeline role:
            Post-localization comparison with gold locations.
        """
        return len(set(predicted[:k]).intersection(gold)) / len(gold)

    return LevelMetrics(
        hit_at_1=float(bool(predicted and predicted[0] in gold)),
        recall_at_5=recall(5),
        recall_at_10=recall(10),
        mrr=1.0 / first if first else 0.0,
        gold_count=len(gold),
        first_relevant_rank=first,
    )


def _candidate_function(candidate: RankedCandidate) -> tuple[str, ...]:
    """
    Purpose:
        Convert a function candidate into the file-and-function identity used for matching.

    Args:
        candidate: One ranked localization candidate.

    Returns:
        A `tuple[str, ...]` result produced by the operation.

    Pipeline role:
        Post-localization comparison with gold locations.
    """
    if not candidate.symbol_name:
        return ()
    path = normalize_repository_path(candidate.file_path)
    names = {candidate.symbol_name.casefold()}
    if candidate.qualified_name:
        parts = candidate.qualified_name.split(".")
        names.add(parts[-1].casefold())
        if len(parts) >= 2:
            names.add(".".join(parts[-2:]).casefold())
    return tuple(f"{path}:{name}" for name in sorted(names))


def evaluate_rankings(
    file_rankings: Sequence[RankedCandidate],
    function_rankings: Sequence[RankedCandidate],
    gold: GoldLocations,
) -> LocalizationMetrics:
    """Evaluate already-produced rankings; gold is accepted only at this later boundary."""
    predicted_files = [normalize_repository_path(item.file_path) for item in file_rankings]
    gold_functions = set(gold.functions)
    predicted_functions: list[str] = []
    for candidate in function_rankings:
        alternatives = _candidate_function(candidate)
        match = next((value for value in alternatives if value in gold_functions), None)
        predicted_functions.append(match or (alternatives[0] if alternatives else ""))
    return LocalizationMetrics(
        file=_level_metrics(predicted_files, set(gold.files)),
        function=_level_metrics(predicted_functions, gold_functions),
    )


def aggregate_metrics(metrics: Sequence[LocalizationMetrics]) -> AggregateMetrics:
    """Macro-average per-instance metrics, including zero values for missing gold."""
    count = len(metrics)
    def mean(values: list[float]) -> float:
        """
        Purpose:
            Return the arithmetic mean, or zero for an empty collection.

        Args:
            values: The values value used by this operation.

        Returns:
            A `float` result produced by the operation.

        Pipeline role:
            Post-localization comparison with gold locations.
        """
        return sum(values) / count if count else 0.0

    return AggregateMetrics(
        instances=count,
        file_hit_at_1=mean([item.file.hit_at_1 for item in metrics]),
        file_recall_at_5=mean([item.file.recall_at_5 for item in metrics]),
        file_recall_at_10=mean([item.file.recall_at_10 for item in metrics]),
        file_mrr=mean([item.file.mrr for item in metrics]),
        function_hit_at_1=mean([item.function.hit_at_1 for item in metrics]),
        function_recall_at_5=mean([item.function.recall_at_5 for item in metrics]),
        function_recall_at_10=mean([item.function.recall_at_10 for item in metrics]),
        function_mrr=mean([item.function.mrr for item in metrics]),
    )
