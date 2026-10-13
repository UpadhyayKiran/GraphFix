"""
Purpose:
    Coordinate dense, hybrid, and static graph localization independently of the BM25 pipeline.

Input:
    GraphFixInput, a method name, optional retrieval configuration, and optional encoder.

Output:
    Gold-free AdvancedLocalizationResult with file and function rankings.

Pipeline stage:
    Gold-free retrieval baselines before evaluation accesses GraphFixGold.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from graphfix.datasets import GraphFixInput
from graphfix.graph import build_repository_graph
from graphfix.parsing import PythonRepositoryParser
from graphfix.pipeline.models import (
    AdvancedLocalizationResult,
    AdvancedRetrievalConfig,
    LocalizationMethod,
)
from graphfix.repositories import RepositoryManager, RepositoryRequest
from graphfix.retrieval import (
    BM25Index,
    DenseIndex,
    EmbeddingCache,
    SentenceTransformerEncoder,
    TextEncoder,
    build_file_documents,
    build_function_documents,
    graph_augmented_ranking,
    reciprocal_rank_fusion,
)

DEFAULT_EMBEDDING_CACHE = Path(__file__).resolve().parents[3] / ".cache" / "graphfix" / "embeddings"


class AdvancedLocalizationPipeline:
    """Simple orchestrator for the three new gold-free localization methods."""

    def __init__(
        self,
        repository_manager: RepositoryManager,
        *,
        config: AdvancedRetrievalConfig | None = None,
        parser: PythonRepositoryParser | None = None,
        encoder: TextEncoder | None = None,
        embedding_cache: EmbeddingCache | None = None,
    ) -> None:
        """Store dependencies; the optional dense model is loaded lazily on first use."""
        self._repositories = repository_manager
        self.config = config or AdvancedRetrievalConfig()
        self._parser = parser or PythonRepositoryParser()
        self._encoder = encoder
        self._embedding_cache = embedding_cache or EmbeddingCache(DEFAULT_EMBEDDING_CACHE)

    def localize(
        self, issue: GraphFixInput, *, method: LocalizationMethod
    ) -> AdvancedLocalizationResult:
        """
        Reconstruct, parse, build documents, and rank without accepting GraphFixGold.

        Dense ranks cosine similarity. Hybrid applies RRF to BM25 and dense ranks.
        Graph uses hybrid top candidates as seeds, follows bounded real graph edges,
        and applies a second RRF fusion. File and function branches remain parallel.
        """
        checkout = self._repositories.reconstruct(RepositoryRequest.from_graphfix_input(issue))
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
        encoder = self._encoder or SentenceTransformerEncoder(self.config.dense_model)
        self._encoder = encoder
        file_embeddings = self._embedding_cache.load_or_encode(
            file_documents,
            encoder,
            repository=issue.repo,
            base_commit=issue.base_commit,
            model_name=self.config.dense_model,
            document_level="file",
        )
        function_embeddings = self._embedding_cache.load_or_encode(
            function_documents,
            encoder,
            repository=issue.repo,
            base_commit=issue.base_commit,
            model_name=self.config.dense_model,
            document_level="function",
        )
        # Hybrid must fuse every available candidate before final Top-K truncation.
        # Standalone Dense still materializes only the requested output depth.
        file_fusion_depth = (
            len(file_documents) if method in {"hybrid", "graph"} else self.config.top_k
        )
        function_fusion_depth = (
            len(function_documents) if method in {"hybrid", "graph"} else self.config.top_k
        )
        dense_files = DenseIndex(
            file_documents, encoder, document_embeddings=file_embeddings
        ).search(
            issue.problem_statement, top_k=file_fusion_depth
        )
        dense_functions = DenseIndex(
            function_documents, encoder, document_embeddings=function_embeddings
        ).search(
            issue.problem_statement, top_k=function_fusion_depth
        )
        file_rankings = dense_files
        function_rankings = dense_functions
        if method in {"hybrid", "graph"}:
            bm25_files = BM25Index(
                file_documents, k1=self.config.bm25_k1, b=self.config.bm25_b
            ).search(issue.problem_statement, top_k=len(file_documents))
            bm25_functions = BM25Index(
                function_documents, k1=self.config.bm25_k1, b=self.config.bm25_b
            ).search(issue.problem_statement, top_k=len(function_documents))
            file_rankings = reciprocal_rank_fusion(
                (bm25_files, dense_files), top_k=self.config.top_k, rrf_k=self.config.rrf_k
            )
            function_rankings = reciprocal_rank_fusion(
                (bm25_functions, dense_functions), top_k=self.config.top_k, rrf_k=self.config.rrf_k
            )
        if method == "graph":
            file_rankings = graph_augmented_ranking(
                file_rankings,
                file_documents,
                graph,
                top_k=self.config.top_k,
                seed_top_k=self.config.graph_seed_top_k,
                depth=self.config.graph_depth,
                rrf_k=self.config.rrf_k,
            )
            function_rankings = graph_augmented_ranking(
                function_rankings,
                function_documents,
                graph,
                top_k=self.config.top_k,
                seed_top_k=self.config.graph_seed_top_k,
                depth=self.config.graph_depth,
                rrf_k=self.config.rrf_k,
            )
        return AdvancedLocalizationResult(
            created_at=datetime.now(UTC),
            method=method,
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
            file_rankings=file_rankings,
            function_rankings=function_rankings,
        )
