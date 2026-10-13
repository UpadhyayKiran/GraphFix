"""Safe, bounded, prompted navigation over an existing RepositoryGraph."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

from graphfix.graph import GraphNode, RepositoryGraph
from graphfix.retrieval import RankedCandidate


class ActionName(StrEnum):
    """Actions supported directly by existing RepositoryGraph operations."""

    INSPECT_FILE = "INSPECT_FILE"
    INSPECT_FUNCTION = "INSPECT_FUNCTION"
    EXPAND_CALLERS = "EXPAND_CALLERS"
    EXPAND_CALLEES = "EXPAND_CALLEES"
    EXPAND_IMPORTS = "EXPAND_IMPORTS"
    EXPAND_INHERITANCE = "EXPAND_INHERITANCE"
    STOP = "STOP"


@dataclass(frozen=True)
class AgentAction:
    """One validated action plus model-validation metadata."""

    name: ActionName
    node_id: str | None = None
    validation_errors: tuple[str, ...] = ()
    invalid_attempts: tuple[ControllerAttempt, ...] = ()
    model_calls: int = 0


@dataclass(frozen=True)
class ControllerAttempt:
    """Bounded audit record for one invalid controller attempt."""

    attempt_number: int
    raw_response: str
    validation_error: str


@dataclass(frozen=True)
class ObservedNode:
    """Compact bounded metadata for one inspected or reached graph node."""

    node_id: str
    node_type: str
    file_path: str | None
    name: str
    relationship: str
    direction: str
    snippet: str


@dataclass(frozen=True)
class AgentObservation:
    """Evidence returned after one safe graph action."""

    action: AgentAction
    node_ids: tuple[str, ...]
    message: str
    nodes: tuple[ObservedNode, ...] = ()


@dataclass(frozen=True)
class AgentState:
    """Bounded gold-free context supplied to the prompted controller."""

    file_candidates: tuple[str, ...]
    function_candidates: tuple[str, ...]
    evidence_node_ids: tuple[str, ...]
    steps_remaining: int
    issue: str = ""
    localization_level: str = "file"
    candidate_metadata: tuple[str, ...] = ()
    evidence_metadata: tuple[str, ...] = ()
    history: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentResult:
    """Bounded trace plus deterministic file/function localization rankings."""

    file_rankings: tuple[RankedCandidate, ...]
    function_rankings: tuple[RankedCandidate, ...]
    observations: tuple[AgentObservation, ...]
    stopped_by_controller: bool
    stopping_reason: str
    model_calls: int
    retry_count: int
    validation_failures: int
    observed_node_ids: tuple[str, ...]


class Controller(Protocol):
    """Controller boundary shared by prompted and deterministic implementations."""

    def choose_action(self, state: AgentState) -> AgentAction:
        """Choose one GraphFix-defined action from the current state."""
        ...


def parse_agent_action(text: str) -> AgentAction:
    """Parse strict JSON shaped as ``{"action": "NAME", "node_id": "optional-id"}``."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Controller response must be valid JSON") from error
    if not isinstance(payload, dict) or set(payload) - {"action", "node_id"}:
        raise ValueError("Controller response contains unsupported fields")
    action_value = payload.get("action")
    if not isinstance(action_value, str):
        raise ValueError("Controller response has an unknown action")
    try:
        name = ActionName(action_value)
    except ValueError as error:
        raise ValueError("Controller response has an unknown action") from error
    node_id = payload.get("node_id")
    if node_id is not None and not isinstance(node_id, str):
        raise ValueError("node_id must be a string")
    if name is ActionName.STOP and node_id is not None:
        raise ValueError("STOP requires null node_id")
    if name is not ActionName.STOP and not node_id:
        raise ValueError("This action requires node_id")
    return AgentAction(name=name, node_id=node_id)


class PromptedController:
    """Strict prompted controller with bounded validation retries and safe STOP."""

    def __init__(
        self,
        generate: Callable[[str], str],
        *,
        max_retries: int = 2,
        validate: Callable[[AgentAction], None] | None = None,
    ) -> None:
        """Configure generation, retry bounds, and optional action validation."""
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._generate = generate
        self.max_retries = max_retries
        self._validate = validate

    @staticmethod
    def _prompt(state: AgentState) -> str:
        """Build the deterministic gold-free controller prompt for one step."""
        payload = {
            "issue": state.issue,
            "localization_level": state.localization_level,
            "initial_candidates": list(state.candidate_metadata),
            "accumulated_evidence": list(state.evidence_metadata),
            "evidence_node_ids": list(state.evidence_node_ids),
            "history": list(state.history),
            "steps_remaining": state.steps_remaining,
        }
        schema = '{"action":"ACTION_NAME","node_id":"graph-node-id-or-null"}'
        if state.localization_level == "file":
            level_guidance = (
                "FILE navigation: candidate targets are file nodes. Use INSPECT_FILE to "
                "inspect a file; never use INSPECT_FUNCTION on a file node. Use a graph "
                "expansion only when it is compatible with the selected node and relation."
            )
        else:
            level_guidance = (
                "FUNCTION navigation: candidate targets are function or method nodes. Use "
                "INSPECT_FUNCTION to inspect them; do not use INSPECT_FILE on a function "
                "or method node. Use a graph expansion only when it is compatible with "
                "the selected node and relation."
            )
        return (
            "Select one safe repository-navigation action using only the gold-free context.\n"
            f"Navigation level: {state.localization_level.upper()}. {level_guidance}\n"
            "Action priority policy:\n"
            "1. Prefer an action likely to add NEW localization evidence: an unobserved "
            "node, unseen relevant relationship, different candidate support, unseen "
            "source, or information that could change or strengthen the ranking.\n"
            "2. Inspect prior action RESULTS in history, not only action names. If the same "
            "action and target already returned no new nodes, relationships, or source, "
            "treat that pair as exhausted and do not repeat it unless new information could "
            "now result. Observed node IDs are evidence already collected; inspecting them "
            "again is normally low value.\n"
            "3. After an exhausted action, try a different promising unobserved candidate "
            "or a different compatible relation. Revisit only for genuinely different, "
            "previously unseen evidence.\n"
            "4. STOP when evidence is adequate, recent actions add no new evidence, all "
            "promising actions would repeat evidence, or no useful compatible unexplored "
            "action remains. Do not consume steps merely because they remain; the hard "
            "step limit is a safety maximum, not a target length.\n"
            "Node-ID rule: for a non-STOP action, copy one exact compatible graph node_id "
            "from an explicit node_id field in the current candidates or collected "
            "evidence. Never invent an ID or derive one from prose, serialized history, "
            "validation errors, source snippets, explanations, or concatenated text.\n"
            'STOP schema: {"action":"STOP","node_id":null}. STOP must ALWAYS use null; '
            "every non-STOP action must use an exact available compatible graph node ID.\n"
            "Review the candidate node IDs, observed node IDs, accumulated evidence, prior "
            "action results, and remaining budget in the context below.\n"
            f"Allowed actions: {', '.join(action.value for action in ActionName)}.\n"
            f"Steps remaining: {state.steps_remaining}.\n"
            f"Required JSON schema: {schema}. Return exactly one JSON object only; STOP "
            "uses null node_id. No prose, Markdown, explanation, reasoning, confidence, or "
            "additional fields.\n"
            f"Context: {json.dumps(payload, sort_keys=True, separators=(',', ':'))}"
        )

    def choose_action(self, state: AgentState) -> AgentAction:
        """Retry invalid output, then return deterministic STOP instead of raising."""
        prompt = self._prompt(state)
        errors: list[str] = []
        invalid_attempts: list[ControllerAttempt] = []
        for attempt in range(self.max_retries + 1):
            try:
                response = self._generate(prompt)
            except RuntimeError as error:
                response = ""
                validation_error = f"Provider error: {error}"
                errors.append(validation_error)
                invalid_attempts.append(
                    ControllerAttempt(attempt + 1, response, validation_error)
                )
                continue
            try:
                action = parse_agent_action(response)
                if self._validate is not None:
                    self._validate(action)
            except ValueError as error:
                errors.append(str(error))
                invalid_attempts.append(
                    ControllerAttempt(attempt + 1, response[:1000], str(error))
                )
                prompt = (
                    f"{self._prompt(state)}\nPrevious response validation error: {error}. "
                    "Return one corrected JSON object only."
                )
                continue
            return replace(
                action,
                validation_errors=tuple(errors),
                invalid_attempts=tuple(invalid_attempts),
                model_calls=attempt + 1,
            )
        return AgentAction(
            ActionName.STOP,
            validation_errors=tuple(errors),
            invalid_attempts=tuple(invalid_attempts),
            model_calls=self.max_retries + 1,
        )


class GraphNavigationEnvironment:
    """Safe state/action/observation wrapper over an existing RepositoryGraph."""

    def __init__(
        self,
        graph: RepositoryGraph,
        file_rankings: Sequence[RankedCandidate],
        function_rankings: Sequence[RankedCandidate],
        *,
        issue: str = "",
        localization_level: str = "file",
        snippets: Mapping[str, str] | None = None,
        max_observed_nodes: int = 8,
        max_snippet_chars: int = 400,
        expand_candidate_pool: bool = False,
        max_expanded_candidates: int = 5,
    ) -> None:
        """Create a bounded navigation view over the graph and initial rankings."""
        if localization_level not in {"file", "function"}:
            raise ValueError("localization_level must be file or function")
        if (
            max_observed_nodes <= 0
            or max_snippet_chars <= 0
            or max_expanded_candidates <= 0
        ):
            raise ValueError("observation bounds must be positive")
        self.graph = graph
        self.file_rankings = tuple(file_rankings)
        self.function_rankings = tuple(function_rankings)
        self.issue = issue
        self.localization_level = localization_level
        self.snippets = dict(snippets or {})
        self.max_observed_nodes = max_observed_nodes
        self.max_snippet_chars = max_snippet_chars
        self.expand_candidate_pool = expand_candidate_pool
        self.max_expanded_candidates = max_expanded_candidates
        self.evidence: set[str] = set()
        self.history: list[AgentObservation] = []

    def _node_summary(self, node: GraphNode) -> str:
        """Serialize one graph node into stable bounded prompt evidence."""
        snippet = self.snippets.get(node.node_id, "")[: self.max_snippet_chars]
        return json.dumps(
            {
                "node_id": node.node_id,
                "type": node.node_type,
                "file_path": node.file_path,
                "name": node.name,
                "snippet": snippet,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def state(self, steps_remaining: int) -> AgentState:
        """Build deterministic bounded issue, candidate, evidence, and history context."""
        candidates = (
            self.file_rankings
            if self.localization_level == "file"
            else self.function_rankings
        )
        candidate_metadata = tuple(
            json.dumps(
                {
                    "rank": item.rank,
                    "node_id": item.document_id,
                    "file_path": item.file_path,
                    "qualified_name": item.qualified_name,
                    "snippet": self.snippets.get(item.document_id, "")[
                        : self.max_snippet_chars
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for item in candidates
        )
        evidence_metadata = tuple(
            self._node_summary(node)
            for node_id in sorted(self.evidence)[: self.max_observed_nodes]
            if (node := self.graph.get_node(node_id)) is not None
        )
        history = tuple(
            json.dumps(
                {
                    "action": item.action.name.value,
                    "node_id": item.action.node_id,
                    "observed": list(item.node_ids),
                    "validation_errors": list(item.action.validation_errors),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for item in self.history
        )
        return AgentState(
            file_candidates=tuple(item.document_id for item in self.file_rankings),
            function_candidates=tuple(item.document_id for item in self.function_rankings),
            evidence_node_ids=tuple(sorted(self.evidence)),
            steps_remaining=steps_remaining,
            issue=self.issue,
            localization_level=self.localization_level,
            candidate_metadata=candidate_metadata,
            evidence_metadata=evidence_metadata,
            history=history,
        )

    def _relationship(self, action: ActionName) -> tuple[str, str]:
        """Map an expansion action to its graph relationship and direction."""
        return {
            ActionName.INSPECT_FILE: ("SELF", "self"),
            ActionName.INSPECT_FUNCTION: ("SELF", "self"),
            ActionName.EXPAND_CALLERS: ("CALLS", "incoming"),
            ActionName.EXPAND_CALLEES: ("CALLS", "outgoing"),
            ActionName.EXPAND_IMPORTS: ("IMPORTS", "outgoing"),
            ActionName.EXPAND_INHERITANCE: ("INHERITS", "outgoing"),
        }[action]

    def step(self, action: AgentAction) -> AgentObservation:
        """Validate and apply one action without executing model-generated code."""
        self.validate_action(action)
        if action.name is ActionName.STOP:
            message = (
                "Safe STOP after invalid controller responses"
                if action.validation_errors
                else "Controller requested STOP"
            )
            observation = AgentObservation(action, (), message)
            self.history.append(observation)
            return observation
        node = self.graph.get_node(action.node_id or "")
        assert node is not None

        reached: tuple[GraphNode, ...]
        if action.name is ActionName.INSPECT_FILE or action.name is ActionName.INSPECT_FUNCTION:
            reached = (node,)
        elif action.name is ActionName.EXPAND_CALLERS:
            reached = self.graph.callers(node.node_id)
        elif action.name is ActionName.EXPAND_CALLEES:
            reached = self.graph.callees(node.node_id)
        elif action.name is ActionName.EXPAND_IMPORTS:
            reached = self.graph.imports(node.node_id)
        else:
            reached = self.graph.bases(node.node_id)

        relationship, direction = self._relationship(action.name)
        bounded = tuple(sorted(reached, key=lambda item: item.node_id))[
            : self.max_observed_nodes
        ]
        details = tuple(
            ObservedNode(
                node_id=item.node_id,
                node_type=item.node_type,
                file_path=item.file_path,
                name=item.name,
                relationship=relationship,
                direction=direction,
                snippet=self.snippets.get(item.node_id, "")[: self.max_snippet_chars],
            )
            for item in bounded
        )
        node_ids = tuple(item.node_id for item in bounded)
        self.evidence.update(node_ids)
        self.evidence.add(node.node_id)
        observation = AgentObservation(
            action,
            node_ids,
            f"Observed {len(node_ids)} graph node(s) through {direction} {relationship}",
            details,
        )
        self.history.append(observation)
        return observation

    def validate_action(self, action: AgentAction) -> None:
        """Reject unknown or type-incompatible graph actions without changing state."""
        if action.name is ActionName.STOP:
            return
        node = self.graph.get_node(action.node_id or "")
        if node is None:
            raise ValueError(f"Unknown graph node: {action.node_id}")
        if action.name is ActionName.INSPECT_FILE:
            if node.node_type != "file":
                raise ValueError("INSPECT_FILE requires a file node")
        elif action.name is ActionName.INSPECT_FUNCTION:
            if node.node_type not in {"function", "method"}:
                raise ValueError("INSPECT_FUNCTION requires a function or method node")
        elif action.name in {ActionName.EXPAND_CALLERS, ActionName.EXPAND_CALLEES}:
            if node.node_type not in {"function", "method"}:
                raise ValueError(f"{action.name} requires a function or method node")
        elif action.name is ActionName.EXPAND_IMPORTS and node.node_type != "file":
            raise ValueError("EXPAND_IMPORTS requires a file node")
        elif action.name is ActionName.EXPAND_INHERITANCE and node.node_type != "class":
            raise ValueError("EXPAND_INHERITANCE requires a class node")

    def reranked_candidates(
        self,
    ) -> tuple[tuple[RankedCandidate, ...], tuple[RankedCandidate, ...]]:
        """Rerank candidates, optionally admitting bounded observed graph nodes."""

        def discovered_candidates(
            candidates: tuple[RankedCandidate, ...], level: str
        ) -> tuple[RankedCandidate, ...]:
            """Convert observed type-compatible graph nodes into bounded candidates."""
            if not self.expand_candidate_pool:
                return ()
            initial_ids = {item.document_id for item in candidates}
            found: dict[str, RankedCandidate] = {}
            for node_id in sorted(self.evidence):
                node = self.graph.get_node(node_id)
                if node is None or node.file_path is None:
                    continue
                if level == "file":
                    document_id = f"file:{node.file_path}"
                    candidate = RankedCandidate(
                        rank=1,
                        score=0.0,
                        document_id=document_id,
                        kind="file",
                        file_path=node.file_path,
                    )
                elif node.node_type in {"function", "method"}:
                    candidate = RankedCandidate(
                        rank=1,
                        score=0.0,
                        document_id=node.node_id,
                        kind=node.node_type,
                        file_path=node.file_path,
                        symbol_name=node.name,
                        qualified_name=node.qualified_name,
                        start_line=node.start_line,
                        end_line=node.end_line,
                    )
                else:
                    continue
                if candidate.document_id not in initial_ids:
                    found.setdefault(candidate.document_id, candidate)
            return tuple(found[key] for key in sorted(found))[
                : self.max_expanded_candidates
            ]

        def rerank(
            candidates: tuple[RankedCandidate, ...], level: str
        ) -> tuple[RankedCandidate, ...]:
            """Promote supported candidates while retaining Hybrid rank as the prior."""
            expanded = discovered_candidates(candidates, level)
            expanded_ids = {item.document_id for item in expanded}
            ordered = sorted(
                (*candidates, *expanded),
                key=lambda item: (
                    not (
                        item.document_id in self.evidence
                        or f"file:{item.file_path}" in self.evidence
                        or item.document_id in expanded_ids
                    ),
                    item.document_id in expanded_ids,
                    item.rank if item.document_id not in expanded_ids else 0,
                    item.document_id,
                ),
            )
            return tuple(
                item.model_copy(update={"rank": rank})
                for rank, item in enumerate(ordered[: len(candidates)], start=1)
            )

        return rerank(self.file_rankings, "file"), rerank(
            self.function_rankings, "function"
        )


class AgentRunner:
    """Run prompted navigation with an explicit fixed step limit."""

    def __init__(self, controller: Controller, *, max_steps: int = 10) -> None:
        """Configure a controller with a positive hard navigation-step limit."""
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.controller = controller
        self.max_steps = max_steps

    def run(self, environment: GraphNavigationEnvironment) -> AgentResult:
        """Repeat state/action/observation until explicit or validation-safe STOP or cap."""
        observations: list[AgentObservation] = []
        stopped = False
        stopping_reason = "max_steps"
        for step in range(self.max_steps):
            action = self.controller.choose_action(
                environment.state(self.max_steps - step)
            )
            try:
                observation = environment.step(action)
            except ValueError as error:
                safe_stop = replace(
                    action,
                    name=ActionName.STOP,
                    node_id=None,
                    validation_errors=(*action.validation_errors, str(error)),
                )
                observation = environment.step(safe_stop)
            observations.append(observation)
            if observation.action.name is ActionName.STOP:
                stopped = True
                stopping_reason = (
                    "invalid_response_stop"
                    if observation.action.validation_errors
                    else "explicit_stop"
                )
                break
        files, functions = environment.reranked_candidates()
        return AgentResult(
            files,
            functions,
            tuple(observations),
            stopped,
            stopping_reason,
            sum(item.action.model_calls for item in observations),
            sum(max(item.action.model_calls - 1, 0) for item in observations),
            sum(len(item.action.validation_errors) for item in observations),
            tuple(sorted(environment.evidence)),
        )
