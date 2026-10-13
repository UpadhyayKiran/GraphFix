"""
Purpose:
    Coordinates the end-to-end BM25 localization pipeline.

Input:
    GraphFixInput; GraphFixInstance only for evaluation.

Output:
    Gold-free localization and optional evaluation.

Pipeline stage:
    BM25 baseline orchestration.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from graphfix.datasets.models import GraphFixInput, GraphFixInstance
from graphfix.evaluation import evaluate_rankings, extract_gold_locations
from graphfix.graph import build_repository_graph
from graphfix.parsing import PythonRepositoryParser
from graphfix.pipeline.models import (
    EvaluationResult,
    LocalizationResult,
    PipelineModel,
    RetrievalConfig,
)
from graphfix.repositories import RepositoryManager, RepositoryRequest
from graphfix.retrieval import BM25Index, build_file_documents, build_function_documents


class BM25LocalizationPipeline:
    """AST/graph/BM25 vertical slice over verified repository checkouts."""

    def __init__(
        self,
        repository_manager: RepositoryManager,
        *,
        config: RetrievalConfig | None = None,
        parser: PythonRepositoryParser | None = None,
    ) -> None:
        """
        Purpose:
            Initialize the object and store the configuration needed by later operations.

        Args:
            repository_manager: The repository manager value used by this operation.
            config: The config value used by this operation.
            parser: The parser value used by this operation.

        Returns:
            Nothing. The operation updates state, writes output, or performs a check directly.

        Pipeline role:
            Review-1 pipeline and resumable campaign execution.
        """
        self._repositories = repository_manager
        self.config = config or RetrievalConfig()
        self._parser = parser or PythonRepositoryParser()

    def localize(self, issue: GraphFixInput) -> LocalizationResult:
        """Rank files/functions using only inference-visible issue data."""
        request = RepositoryRequest.from_graphfix_input(issue)
        checkout = self._repositories.reconstruct(request)
        if not checkout.success or not checkout.checkout_path or not checkout.resolved_commit:
            message = (
                checkout.failure.message if checkout.failure else "unknown reconstruction failure"
            )
            raise RuntimeError(f"Unable to reconstruct {issue.instance_id}: {message}")
        if checkout.resolved_commit.casefold() != issue.base_commit.casefold():
            raise RuntimeError("Verified checkout commit does not match the requested base_commit")

        parsed = self._parser.parse(Path(checkout.checkout_path))
        graph = build_repository_graph(parsed, repository_name=issue.repo)
        file_documents = build_file_documents(
            parsed, max_source_chars=self.config.max_file_source_chars
        )
        function_documents = build_function_documents(
            parsed, max_source_chars=self.config.max_function_source_chars
        )
        file_index = BM25Index(
            file_documents, k1=self.config.bm25_k1, b=self.config.bm25_b
        )
        function_index = BM25Index(
            function_documents, k1=self.config.bm25_k1, b=self.config.bm25_b
        )
        return LocalizationResult(
            created_at=datetime.now(UTC),
            instance_id=issue.instance_id,
            repository=issue.repo,
            base_commit=issue.base_commit,
            checked_out_commit=checkout.resolved_commit,
            checkout_path=checkout.checkout_path,
            issue=issue.problem_statement,
            parser_stats=parsed.stats,
            parser_failures=parsed.failures,
            graph_stats=graph.stats,
            retrieval_config=self.config,
            file_rankings=file_index.search(issue.problem_statement, top_k=self.config.top_k),
            function_rankings=function_index.search(
                issue.problem_statement, top_k=self.config.top_k
            ),
        )

    def evaluate(self, instance: GraphFixInstance) -> EvaluationResult:
        """Run gold-free localization first, then evaluate the completed rankings."""
        localization = self.localize(instance.input)
        gold = extract_gold_locations(instance.gold)
        metrics = evaluate_rankings(
            localization.file_rankings,
            localization.function_rankings,
            gold,
        )
        return EvaluationResult(localization=localization, gold=gold, metrics=metrics)


# Temporary compatibility alias for existing integrations.
Review1Pipeline = BM25LocalizationPipeline


def save_result(result: PipelineModel, path: Path) -> None:
    """Persist a structured result as readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
