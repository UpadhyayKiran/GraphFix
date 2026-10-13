"""
Purpose:
    Defines configuration and serializable pipeline results.

Input:
    Stage outputs, rankings, gold, and metrics.

Output:
    LocalizationResult and EvaluationResult.

Pipeline stage:
    Pipeline contracts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from graphfix.evaluation import GoldLocations, LocalizationMetrics
from graphfix.graph import GraphStats
from graphfix.parsing import ParserStats
from graphfix.retrieval import RankedCandidate


class PipelineModel(BaseModel):
    """
    Purpose:
        Base model that rejects unexpected pipeline result fields.

    Pipeline role:
        Localization pipeline and resumable campaign execution.
    """
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RetrievalConfig(PipelineModel):
    """
    Purpose:
        Stores top_k and the Okapi BM25 k1 and b settings.

    Contains:
        top_k: The top k value.
        bm25_k1: The bm25 k1 value.
        bm25_b: The bm25 b value.
        max_file_source_chars: The max file source chars value.
        max_function_source_chars: The max function source chars value.

    Pipeline role:
        Localization pipeline and resumable campaign execution.
    """
    top_k: int = Field(default=10, ge=1)
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    max_file_source_chars: int = Field(default=12_000, ge=1)
    max_function_source_chars: int = Field(default=8_000, ge=1)


class LocalizationResult(PipelineModel):
    """Inference result; deliberately contains no gold fields or metrics."""

    schema_version: int = 1
    created_at: datetime
    instance_id: str
    repository: str
    base_commit: str
    checked_out_commit: str
    checkout_path: str
    issue: str
    parser_stats: ParserStats
    parser_failures: tuple[str, ...]
    graph_stats: GraphStats
    retrieval_config: RetrievalConfig
    file_rankings: tuple[RankedCandidate, ...]
    function_rankings: tuple[RankedCandidate, ...]


class EvaluationResult(PipelineModel):
    """Post-inference result that adds evaluation-only gold and metrics."""

    localization: LocalizationResult
    gold: GoldLocations
    metrics: LocalizationMetrics


LocalizationMethod = Literal["dense", "hybrid", "graph"]


class AdvancedRetrievalConfig(PipelineModel):
    """Small configuration for advanced retrieval baselines."""

    top_k: int = Field(default=10, ge=1)
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rrf_k: int = Field(default=60, ge=1)
    graph_seed_top_k: int = Field(default=10, ge=1)
    graph_depth: int = Field(default=1, ge=1)
    max_file_source_chars: int = Field(default=12_000, ge=1)
    max_function_source_chars: int = Field(default=8_000, ge=1)


class AdvancedLocalizationResult(PipelineModel):
    """Gold-free dense, hybrid, or static-graph localization result."""

    schema_version: int = 1
    created_at: datetime
    method: LocalizationMethod
    instance_id: str
    repository: str
    base_commit: str
    checked_out_commit: str
    checkout_path: str
    issue: str
    parser_stats: ParserStats
    parser_failures: tuple[str, ...]
    graph_stats: GraphStats
    retrieval_config: AdvancedRetrievalConfig
    file_rankings: tuple[RankedCandidate, ...]
    function_rankings: tuple[RankedCandidate, ...]


class AdaptiveNavigationConfig(PipelineModel):
    """Bounded prompted-navigation and model-adapter configuration."""

    provider: str = "ollama"
    model: str
    endpoint: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    seed: int = 0
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_steps: int = Field(default=10, ge=1)
    max_retries: int = Field(default=2, ge=0)
    max_observed_nodes: int = Field(default=8, ge=1)
    max_snippet_chars: int = Field(default=400, ge=1)
    candidate_pool: Literal["fixed", "expanded"] = "fixed"
    max_expanded_candidates: int = Field(default=5, ge=1)


class AdaptiveObservedNode(PipelineModel):
    """Serializable compact graph evidence exposed to the controller."""

    node_id: str
    node_type: str
    file_path: str | None
    name: str
    relationship: str
    direction: str
    snippet: str


class AdaptiveTraceStep(PipelineModel):
    """One controller action, validation record, and graph observation."""

    action: str
    node_id: str | None
    observed_node_ids: tuple[str, ...]
    observed_nodes: tuple[AdaptiveObservedNode, ...]
    message: str
    validation_errors: tuple[str, ...]
    invalid_attempts: tuple[AdaptiveControllerAttempt, ...]
    model_calls: int


class AdaptiveControllerAttempt(PipelineModel):
    """Bounded invalid controller response retained for auditability."""

    attempt_number: int
    raw_response: str
    validation_error: str


class AdaptiveLevelResult(PipelineModel):
    """One deterministic file- or function-level navigation episode."""

    localization_level: Literal["file", "function"]
    initial_ranking: tuple[RankedCandidate, ...]
    final_ranking: tuple[RankedCandidate, ...]
    trace: tuple[AdaptiveTraceStep, ...]
    max_steps: int
    actual_steps: int
    model_calls: int
    retry_count: int
    validation_failures: int
    observed_node_ids: tuple[str, ...]
    expanded_candidate_ids: tuple[str, ...] = ()
    stopped_by_controller: bool
    stopping_reason: Literal["explicit_stop", "invalid_response_stop", "max_steps"]


class AdaptiveLocalizationResult(PipelineModel):
    """Serializable gold-free result for two prompted navigation episodes."""

    schema_version: int = 1
    created_at: datetime
    method: Literal["adaptive"] = "adaptive"
    instance_id: str
    repository: str
    base_commit: str
    checked_out_commit: str
    checkout_path: str
    issue: str
    parser_stats: ParserStats
    parser_failures: tuple[str, ...]
    graph_stats: GraphStats
    adaptive_config: AdaptiveNavigationConfig
    file_navigation: AdaptiveLevelResult
    function_navigation: AdaptiveLevelResult
    file_rankings: tuple[RankedCandidate, ...]
    function_rankings: tuple[RankedCandidate, ...]
    runtime_seconds: float = Field(ge=0.0)
