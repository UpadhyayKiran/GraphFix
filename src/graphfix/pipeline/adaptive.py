"""Gold-free end-to-end prompted Adaptive Navigation localization pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from graphfix.agent import (
    AgentResult,
    AgentRunner,
    GraphNavigationEnvironment,
    PromptedController,
    TextGenerator,
)
from graphfix.datasets import GraphFixInput
from graphfix.graph import RepositoryGraph, build_repository_graph
from graphfix.parsing import ParsedRepository, PythonRepositoryParser
from graphfix.pipeline.advanced import AdvancedLocalizationPipeline
from graphfix.pipeline.models import (
    AdaptiveControllerAttempt,
    AdaptiveLevelResult,
    AdaptiveLocalizationResult,
    AdaptiveNavigationConfig,
    AdaptiveObservedNode,
    AdaptiveTraceStep,
    AdvancedRetrievalConfig,
)
from graphfix.repositories import RepositoryManager, RepositoryRequest
from graphfix.retrieval import RankedCandidate


def _snippets(parsed: ParsedRepository) -> dict[str, str]:
    """Map graph-compatible document IDs to bounded source supplied by the parser."""
    snippets = {
        f"file:{parsed_file.file_path}": parsed_file.source
        for parsed_file in parsed.files
    }
    snippets.update({symbol.symbol_id: symbol.source for symbol in parsed.symbols})
    return snippets


def _level_result(
    level: str,
    initial: tuple[RankedCandidate, ...],
    result: AgentResult,
    config: AdaptiveNavigationConfig,
) -> AdaptiveLevelResult:
    """Convert one navigation episode into its persisted level result and trace."""
    final = result.file_rankings if level == "file" else result.function_rankings
    trace = tuple(
        AdaptiveTraceStep(
            action=observation.action.name.value,
            node_id=observation.action.node_id,
            observed_node_ids=observation.node_ids,
            observed_nodes=tuple(
                AdaptiveObservedNode(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    file_path=node.file_path,
                    name=node.name,
                    relationship=node.relationship,
                    direction=node.direction,
                    snippet=node.snippet,
                )
                for node in observation.nodes
            ),
            message=observation.message,
            validation_errors=observation.action.validation_errors,
            invalid_attempts=tuple(
                AdaptiveControllerAttempt(
                    attempt_number=attempt.attempt_number,
                    raw_response=attempt.raw_response,
                    validation_error=attempt.validation_error,
                )
                for attempt in observation.action.invalid_attempts
            ),
            model_calls=observation.action.model_calls,
        )
        for observation in result.observations
    )
    initial_ids = {candidate.document_id for candidate in initial}
    return AdaptiveLevelResult(
        localization_level=level,  # type: ignore[arg-type]
        initial_ranking=initial,
        final_ranking=final,
        trace=trace,
        max_steps=config.max_steps,
        actual_steps=len(trace),
        model_calls=result.model_calls,
        retry_count=result.retry_count,
        validation_failures=result.validation_failures,
        observed_node_ids=result.observed_node_ids,
        expanded_candidate_ids=tuple(
            candidate.document_id
            for candidate in final
            if candidate.document_id not in initial_ids
        ),
        stopped_by_controller=result.stopped_by_controller,
        stopping_reason=result.stopping_reason,  # type: ignore[arg-type]
    )


class AdaptiveNavigationPipeline:
    """Connect corrected Hybrid retrieval to two bounded prompted graph episodes."""

    def __init__(
        self,
        repository_manager: RepositoryManager,
        generator: TextGenerator,
        *,
        config: AdaptiveNavigationConfig,
        parser: PythonRepositoryParser | None = None,
        hybrid_pipeline: AdvancedLocalizationPipeline | None = None,
    ) -> None:
        """Configure repository, Hybrid, parser, controller, and traversal dependencies."""
        self._repositories = repository_manager
        self._generator = generator
        self.config = config
        self._parser = parser or PythonRepositoryParser()
        self._hybrid = hybrid_pipeline or AdvancedLocalizationPipeline(
            repository_manager,
            config=AdvancedRetrievalConfig(top_k=10),
        )

    def _navigate(
        self,
        *,
        issue: GraphFixInput,
        graph: RepositoryGraph,
        snippets: dict[str, str],
        files: tuple[RankedCandidate, ...],
        functions: tuple[RankedCandidate, ...],
        level: str,
    ) -> AgentResult:
        """Run one bounded file- or function-level gold-free navigation episode."""
        environment = GraphNavigationEnvironment(
            graph,
            files,
            functions,
            issue=issue.problem_statement,
            localization_level=level,
            snippets=snippets,
            max_observed_nodes=self.config.max_observed_nodes,
            max_snippet_chars=self.config.max_snippet_chars,
            expand_candidate_pool=self.config.candidate_pool == "expanded",
            max_expanded_candidates=self.config.max_expanded_candidates,
        )
        controller = PromptedController(
            self._generator,
            max_retries=self.config.max_retries,
            validate=environment.validate_action,
        )
        return AgentRunner(controller, max_steps=self.config.max_steps).run(environment)

    def localize(self, issue: GraphFixInput) -> AdaptiveLocalizationResult:
        """Produce final rankings and traces without accepting or accessing gold."""
        started = perf_counter()
        hybrid = self._hybrid.localize(issue, method="hybrid")
        checkout = self._repositories.reconstruct(
            RepositoryRequest.from_graphfix_input(issue)
        )
        if not checkout.success or not checkout.checkout_path or not checkout.resolved_commit:
            message = checkout.failure.message if checkout.failure else "unknown failure"
            raise RuntimeError(f"Unable to reconstruct {issue.instance_id}: {message}")
        if checkout.resolved_commit.casefold() != issue.base_commit.casefold():
            raise RuntimeError("Verified checkout commit does not match base_commit")
        parsed = self._parser.parse(Path(checkout.checkout_path))
        graph = build_repository_graph(parsed, repository_name=issue.repo)
        snippets = _snippets(parsed)

        file_agent = self._navigate(
            issue=issue,
            graph=graph,
            snippets=snippets,
            files=hybrid.file_rankings,
            functions=hybrid.function_rankings,
            level="file",
        )
        function_agent = self._navigate(
            issue=issue,
            graph=graph,
            snippets=snippets,
            files=hybrid.file_rankings,
            functions=hybrid.function_rankings,
            level="function",
        )
        file_navigation = _level_result(
            "file", hybrid.file_rankings, file_agent, self.config
        )
        function_navigation = _level_result(
            "function", hybrid.function_rankings, function_agent, self.config
        )
        return AdaptiveLocalizationResult(
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
            adaptive_config=self.config,
            file_navigation=file_navigation,
            function_navigation=function_navigation,
            file_rankings=file_navigation.final_ranking,
            function_rankings=function_navigation.final_ranking,
            runtime_seconds=perf_counter() - started,
        )
