"""Gold-free live localization for a pinned public GitHub Python repository."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

from graphfix.datasets import GraphFixInput
from graphfix.graph import RepositoryGraph, build_repository_graph
from graphfix.parsing import ParsedRepository, PythonRepositoryParser
from graphfix.pipeline.models import AdvancedRetrievalConfig
from graphfix.repositories import RepositoryManager, RepositoryRequest
from graphfix.repositories.git import GitRunner, run_git
from graphfix.retrieval import (
    BM25Index,
    DenseIndex,
    EmbeddingCache,
    RankedCandidate,
    SearchDocument,
    TextEncoder,
    build_file_documents,
    build_function_documents,
    graph_augmented_ranking,
    reciprocal_rank_fusion,
)

_FULL_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_public_github_url(value: str) -> str:
    """Return ``owner/repository`` for a canonical public GitHub HTTPS URL."""
    parsed = urlparse(value.strip())
    if parsed.scheme != "https" or parsed.hostname not in {"github.com", "www.github.com"}:
        raise ValueError("Use an HTTPS public GitHub URL such as https://github.com/owner/repo")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("GitHub URL must not contain credentials, a query, or a fragment")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) != 2:
        raise ValueError("GitHub URL must identify exactly one owner/repository")
    owner, name = parts
    if name.lower().endswith(".git"):
        name = name[:-4]
    if not owner or not name or not _PART.fullmatch(owner) or not _PART.fullmatch(name):
        raise ValueError("GitHub owner or repository name is invalid")
    return f"{owner}/{name}"


def validate_commit(value: str) -> str:
    """Validate and normalize a full Git commit SHA."""
    commit = value.strip().lower()
    if not _FULL_SHA.fullmatch(commit):
        raise ValueError("Commit must be a full 40-character hexadecimal SHA")
    return commit


def resolve_default_branch_head(
    repository_url: str,
    *,
    git_runner: GitRunner = run_git,
    timeout_seconds: float = 30.0,
) -> tuple[str, str, str]:
    """Resolve the public repository's default branch HEAD exactly once."""
    repository = parse_public_github_url(repository_url)
    canonical = f"https://github.com/{repository}.git"
    result = git_runner(
        ["ls-remote", "--symref", canonical, "HEAD"], timeout_seconds=timeout_seconds
    )
    branch: str | None = None
    commit: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("ref: refs/heads/") and line.endswith("\tHEAD"):
            branch = line.removeprefix("ref: refs/heads/").removesuffix("\tHEAD")
        elif line.endswith("\tHEAD"):
            commit = line.split("\t", 1)[0]
    if branch is None or commit is None:
        raise RuntimeError("Git did not return a default branch and HEAD commit")
    return repository, validate_commit(commit), branch


def live_input(repository: str, commit: str, title: str, description: str) -> GraphFixInput:
    """Build an inference-only input; this API deliberately has no gold parameter."""
    issue = "\n\n".join(part.strip() for part in (title, description) if part.strip())
    if not issue:
        raise ValueError("Issue title or description must be non-empty")
    commit = validate_commit(commit)
    identity = hashlib.sha256(f"{repository}\0{commit}\0{issue}".encode()).hexdigest()[:16]
    return GraphFixInput(
        instance_id=f"live-{identity}",
        repo=repository,
        base_commit=commit,
        problem_statement=issue,
        category="live-public-github",
    )


@dataclass(frozen=True)
class PreparedLiveRepository:
    """Reusable in-memory representation of one exact repository commit."""

    issue: GraphFixInput
    checkout_path: Path
    cache_reused: bool
    parsed: ParsedRepository
    graph: RepositoryGraph
    file_documents: tuple[SearchDocument, ...]
    function_documents: tuple[SearchDocument, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class LiveMethodResult:
    """One freshly computed gold-free method result."""

    method: str
    file_rankings: tuple[RankedCandidate, ...]
    function_rankings: tuple[RankedCandidate, ...]
    elapsed_seconds: float


class LiveLocalizationService:
    """Prepare and rank arbitrary public GitHub Python repositories without gold."""

    def __init__(
        self,
        cache_root: Path,
        *,
        repository_manager: RepositoryManager | None = None,
        parser: PythonRepositoryParser | None = None,
        config: AdvancedRetrievalConfig | None = None,
        embedding_cache: EmbeddingCache | None = None,
    ) -> None:
        self._repositories = repository_manager or RepositoryManager(cache_root)
        self._parser = parser or PythonRepositoryParser()
        self.config = config or AdvancedRetrievalConfig()
        self._embeddings = embedding_cache or EmbeddingCache(
            Path(__file__).resolve().parents[2] / ".cache" / "graphfix" / "live_embeddings"
        )

    def prepare(self, issue: GraphFixInput) -> PreparedLiveRepository:
        """Checkout, parse, and build graph/documents without executing target code."""
        started = perf_counter()
        checkout = self._repositories.reconstruct(RepositoryRequest.from_graphfix_input(issue))
        if not checkout.success or checkout.checkout_path is None:
            message = checkout.failure.message if checkout.failure else "unknown checkout failure"
            raise RuntimeError(message)
        parsed = self._parser.parse(Path(checkout.checkout_path))
        if not parsed.files:
            raise ValueError("The pinned repository contains no supported parseable Python files")
        graph = build_repository_graph(parsed, repository_name=issue.repo)
        files = build_file_documents(parsed, max_source_chars=self.config.max_file_source_chars)
        functions = build_function_documents(
            parsed, max_source_chars=self.config.max_function_source_chars
        )
        return PreparedLiveRepository(
            issue=issue,
            checkout_path=Path(checkout.checkout_path),
            cache_reused=checkout.cache_reused,
            parsed=parsed,
            graph=graph,
            file_documents=files,
            function_documents=functions,
            elapsed_seconds=perf_counter() - started,
        )

    def bm25(self, prepared: PreparedLiveRepository) -> LiveMethodResult:
        """Compute the existing BM25 baseline over the prepared live repository."""
        started = perf_counter()
        query = prepared.issue.problem_statement
        return LiveMethodResult(
            method="bm25",
            file_rankings=BM25Index(prepared.file_documents).search(query, top_k=self.config.top_k),
            function_rankings=BM25Index(prepared.function_documents).search(
                query, top_k=self.config.top_k
            ),
            elapsed_seconds=perf_counter() - started,
        )

    def dense_hybrid_graph(
        self,
        prepared: PreparedLiveRepository,
        encoder: TextEncoder,
    ) -> tuple[LiveMethodResult, LiveMethodResult, LiveMethodResult]:
        """Compute existing Dense, Hybrid, and Static Graph algorithms once per corpus."""
        started = perf_counter()
        query = prepared.issue.problem_statement
        files = prepared.file_documents
        functions = prepared.function_documents
        model = self.config.dense_model
        file_vectors = self._embeddings.load_or_encode(
            files, encoder, repository=prepared.issue.repo, base_commit=prepared.issue.base_commit,
            model_name=model, document_level="file",
        )
        function_vectors = self._embeddings.load_or_encode(
            functions, encoder, repository=prepared.issue.repo,
            base_commit=prepared.issue.base_commit, model_name=model, document_level="function",
        )
        dense_files_full = DenseIndex(files, encoder, document_embeddings=file_vectors).search(
            query, top_k=len(files)
        )
        dense_functions_full = DenseIndex(
            functions, encoder, document_embeddings=function_vectors
        ).search(query, top_k=len(functions))
        dense_elapsed = perf_counter() - started
        dense = LiveMethodResult(
            "dense", dense_files_full[: self.config.top_k],
            dense_functions_full[: self.config.top_k], dense_elapsed,
        )
        hybrid_started = perf_counter()
        bm25_files = BM25Index(files).search(query, top_k=len(files))
        bm25_functions = BM25Index(functions).search(query, top_k=len(functions))
        hybrid_files = reciprocal_rank_fusion(
            (bm25_files, dense_files_full), top_k=self.config.top_k, rrf_k=self.config.rrf_k
        )
        hybrid_functions = reciprocal_rank_fusion(
            (bm25_functions, dense_functions_full), top_k=self.config.top_k,
            rrf_k=self.config.rrf_k,
        )
        hybrid = LiveMethodResult(
            "hybrid", hybrid_files, hybrid_functions, perf_counter() - hybrid_started
        )
        graph_started = perf_counter()
        graph = LiveMethodResult(
            "graph",
            graph_augmented_ranking(
                hybrid_files, files, prepared.graph, top_k=self.config.top_k,
                seed_top_k=self.config.graph_seed_top_k, depth=self.config.graph_depth,
                rrf_k=self.config.rrf_k,
            ),
            graph_augmented_ranking(
                hybrid_functions, functions, prepared.graph, top_k=self.config.top_k,
                seed_top_k=self.config.graph_seed_top_k, depth=self.config.graph_depth,
                rrf_k=self.config.rrf_k,
            ),
            perf_counter() - graph_started,
        )
        return dense, hybrid, graph


def save_live_artifact(
    output_root: Path,
    repository_url: str,
    branch: str,
    prepared: PreparedLiveRepository,
    methods: dict[str, LiveMethodResult],
    *,
    adaptive: dict[str, Any] | None = None,
) -> Path:
    """Persist live output separately and atomically; never writes frozen result trees."""
    timestamp = datetime.now(UTC)
    destination = output_root / prepared.issue.instance_id / "localization.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "mode": "live_public_github",
        "created_at": timestamp.isoformat(),
        "repository_url": repository_url,
        "repository": prepared.issue.repo,
        "default_branch": branch,
        "resolved_commit": prepared.issue.base_commit,
        "issue": prepared.issue.model_dump(mode="json"),
        "preparation": {
            "checkout_path": str(prepared.checkout_path),
            "cache_reused": prepared.cache_reused,
            "elapsed_seconds": prepared.elapsed_seconds,
            "python_files": prepared.parsed.stats.python_files_parsed,
            "graph_nodes": prepared.graph.stats.total_nodes,
            "graph_edges": prepared.graph.stats.total_edges,
        },
        "methods": {
            name: {
                "elapsed_seconds": result.elapsed_seconds,
                "file_rankings": [item.model_dump(mode="json") for item in result.file_rankings],
                "function_rankings": [
                    item.model_dump(mode="json") for item in result.function_rankings
                ],
            }
            for name, result in methods.items()
        },
        "adaptive": adaptive,
        "gold": None,
        "metrics": None,
    }
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination
