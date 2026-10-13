import json
from unittest.mock import patch

from graphfix.agent import (
    AGENT_ACTION_SCHEMA,
    ActionName,
    AgentAction,
    AgentRunner,
    AgentState,
    GraphNavigationEnvironment,
    OllamaConfig,
    OllamaTextGenerator,
    PromptedController,
    parse_agent_action,
)
from graphfix.graph import GraphEdge, GraphNode, RepositoryGraph
from graphfix.retrieval import RankedCandidate


def _candidate() -> RankedCandidate:
    return RankedCandidate(
        rank=1,
        score=1.0,
        document_id="file:a.py",
        kind="file",
        file_path="a.py",
    )


def _environment() -> GraphNavigationEnvironment:
    graph = RepositoryGraph(
        nodes=(GraphNode(node_id="file:a.py", node_type="file", name="a.py", file_path="a.py"),),
        edges=(),
    )
    return GraphNavigationEnvironment(graph, (_candidate(),), ())


def test_action_parser_accepts_schema_and_rejects_arbitrary_text() -> None:
    assert parse_agent_action('{"action":"INSPECT_FILE","node_id":"file:a.py"}') == AgentAction(
        ActionName.INSPECT_FILE, "file:a.py"
    )
    for invalid in (
        "__import__('os').system('whoami')",
        '{"action":"EXECUTE","node_id":"file:a.py"}',
        '{"action":"STOP","code":"print(1)"}',
    ):
        try:
            parse_agent_action(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe action text was accepted")
    for invalid in (
        '{"action":"STOP","node_id":"file:a.py"}',
        '{"action":"INSPECT_FILE","node_id":null}',
    ):
        try:
            parse_agent_action(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("semantically invalid action was accepted")


class _Response:
    def __init__(self, body: str) -> None:
        self._body = body.encode()

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_ollama_request_uses_agent_action_structured_output_schema() -> None:
    captured: dict[str, object] = {}

    def open_request(request: object, *, timeout: float) -> _Response:
        captured["payload"] = json.loads(request.data)  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return _Response(
            '{"response":"{\\"action\\":\\"STOP\\",\\"node_id\\":null}"}'
        )

    generator = OllamaTextGenerator(OllamaConfig(model="test", temperature=0.0))
    with patch("urllib.request.urlopen", open_request):
        assert generator("prompt") == '{"action":"STOP","node_id":null}'
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["format"] == AGENT_ACTION_SCHEMA
    assert payload["options"] == {"temperature": 0.0, "seed": 0}


def test_ollama_rejects_malformed_provider_envelope() -> None:
    generator = OllamaTextGenerator(OllamaConfig(model="test"))
    with patch("urllib.request.urlopen", lambda *args, **kwargs: _Response("not-json")):
        try:
            generator("prompt")
        except RuntimeError as error:
            assert "invalid generation response" in str(error)
        else:
            raise AssertionError("malformed provider response was accepted")


def test_prompted_controller_parses_only_structured_action() -> None:
    controller = PromptedController(
        lambda prompt: '{"action":"INSPECT_FILE","node_id":"file:a.py"}'
    )
    action = controller.choose_action(AgentState((), (), (), 1))
    assert action.name is ActionName.INSPECT_FILE


def test_controller_prompt_makes_level_novelty_history_and_stop_explicit() -> None:
    state = AgentState(
        file_candidates=("file:a.py",),
        function_candidates=("function:a.py:a.target:1",),
        evidence_node_ids=("file:seen.py",),
        steps_remaining=4,
        issue="target behavior is incorrect",
        localization_level="file",
        candidate_metadata=("candidate metadata",),
        evidence_metadata=("previously returned evidence",),
        history=("INSPECT_FILE file:seen.py",),
    )
    prompt = PromptedController._prompt(state)
    assert "Navigation level: FILE" in prompt
    assert "Use INSPECT_FILE" in prompt
    assert "never use INSPECT_FUNCTION on a file node" in prompt
    assert "NEW localization evidence" in prompt
    assert "same action and target" in prompt
    assert "treat that pair as exhausted" in prompt
    assert "prior action RESULTS" in prompt
    assert "unobserved candidate" in prompt
    assert "inspecting them again is normally low value" in prompt
    assert "observed node IDs" in prompt
    assert "prior action results" in prompt
    assert "file:seen.py" in prompt
    assert "previously returned evidence" in prompt
    assert "Steps remaining: 4" in prompt
    assert "STOP when evidence is adequate" in prompt
    assert "safety maximum, not a target length" in prompt
    assert "copy one exact compatible graph node_id" in prompt
    assert "Never invent an ID or derive one from prose, serialized history" in prompt
    assert 'STOP schema: {"action":"STOP","node_id":null}' in prompt
    assert "STOP must ALWAYS use null" in prompt
    assert "No prose, Markdown, explanation, reasoning, confidence" in prompt


def test_controller_prompt_uses_function_compatible_inspection_guidance() -> None:
    prompt = PromptedController._prompt(
        AgentState((), (), (), 2, localization_level="function")
    )
    assert "Navigation level: FUNCTION" in prompt
    assert "Use INSPECT_FUNCTION" in prompt
    assert "do not use INSPECT_FILE on a function or method node" in prompt


def test_prompted_controller_retries_then_succeeds() -> None:
    responses = iter(("not-json", '{"action":"STOP","node_id":null}'))
    controller = PromptedController(lambda prompt: next(responses), max_retries=2)
    action = controller.choose_action(AgentState((), (), (), 1, issue="bug"))
    assert action.name is ActionName.STOP
    assert action.model_calls == 2
    assert action.validation_errors == ("Controller response must be valid JSON",)
    assert action.invalid_attempts[0].attempt_number == 1
    assert action.invalid_attempts[0].raw_response == "not-json"


def test_prompted_controller_repeated_invalid_output_safely_stops() -> None:
    controller = PromptedController(lambda prompt: "invalid", max_retries=2)
    action = controller.choose_action(AgentState((), (), (), 1))
    assert action.name is ActionName.STOP
    assert action.model_calls == 3
    assert len(action.validation_errors) == 3
    assert [attempt.attempt_number for attempt in action.invalid_attempts] == [1, 2, 3]


def test_prompted_controller_retries_provider_error_then_succeeds() -> None:
    calls = 0

    def generate(prompt: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("malformed envelope")
        return '{"action":"STOP","node_id":null}'

    action = PromptedController(generate, max_retries=2).choose_action(
        AgentState((), (), (), 1)
    )
    assert action.model_calls == 2
    assert action.validation_errors == ("Provider error: malformed envelope",)
    assert action.invalid_attempts[0].raw_response == ""


def test_prompted_controller_retries_semantically_invalid_action() -> None:
    responses = iter(
        (
            '{"action":"INSPECT_FUNCTION","node_id":"file:a.py"}',
            '{"action":"INSPECT_FILE","node_id":"file:a.py"}',
        )
    )
    environment = _environment()
    action = PromptedController(
        lambda prompt: next(responses),
        max_retries=2,
        validate=environment.validate_action,
    ).choose_action(AgentState((), (), (), 1))
    assert action.name is ActionName.INSPECT_FILE
    assert action.model_calls == 2
    assert action.validation_errors == (
        "INSPECT_FUNCTION requires a function or method node",
    )
    assert action.invalid_attempts[0].raw_response.startswith(
        '{"action":"INSPECT_FUNCTION"'
    )


def test_every_navigation_action_uses_expected_direction() -> None:
    nodes = (
        GraphNode(node_id="file:a.py", node_type="file", name="a.py", file_path="a.py"),
        GraphNode(node_id="file:b.py", node_type="file", name="b.py", file_path="b.py"),
        GraphNode(
            node_id="function:a.f:1", node_type="function", name="f", file_path="a.py"
        ),
        GraphNode(
            node_id="function:b.g:1", node_type="function", name="g", file_path="b.py"
        ),
        GraphNode(node_id="class:a.C:2", node_type="class", name="C", file_path="a.py"),
        GraphNode(node_id="class:b.B:2", node_type="class", name="B", file_path="b.py"),
    )
    graph = RepositoryGraph(
        nodes=nodes,
        edges=(
            GraphEdge(source="function:a.f:1", target="function:b.g:1", edge_type="CALLS"),
            GraphEdge(source="file:a.py", target="file:b.py", edge_type="IMPORTS"),
            GraphEdge(source="class:a.C:2", target="class:b.B:2", edge_type="INHERITS"),
        ),
    )
    environment = GraphNavigationEnvironment(graph, (_candidate(),), ())
    cases = (
        (ActionName.INSPECT_FILE, "file:a.py", "self"),
        (ActionName.INSPECT_FUNCTION, "function:a.f:1", "self"),
        (ActionName.EXPAND_CALLERS, "function:b.g:1", "incoming"),
        (ActionName.EXPAND_CALLEES, "function:a.f:1", "outgoing"),
        (ActionName.EXPAND_IMPORTS, "file:a.py", "outgoing"),
        (ActionName.EXPAND_INHERITANCE, "class:a.C:2", "outgoing"),
    )
    for action, node_id, direction in cases:
        observation = environment.step(AgentAction(action, node_id))
        assert observation.nodes
        assert observation.nodes[0].direction == direction


class NeverStopController:
    def choose_action(self, state: AgentState) -> AgentAction:
        return AgentAction(ActionName.INSPECT_FILE, "file:a.py")


class StopController:
    def choose_action(self, state: AgentState) -> AgentAction:
        return AgentAction(ActionName.STOP)


def test_runner_forces_max_steps_and_supports_explicit_stop() -> None:
    limited = AgentRunner(NeverStopController(), max_steps=2).run(_environment())
    assert len(limited.observations) == 2
    assert not limited.stopped_by_controller
    assert limited.file_rankings[0].document_id == "file:a.py"
    assert limited.retry_count == 0
    assert limited.validation_failures == 0

    stopped = AgentRunner(StopController(), max_steps=10).run(_environment())
    assert len(stopped.observations) == 1
    assert stopped.stopped_by_controller


def _expansion_environment(
    *, expanded: bool, cap: int = 2
) -> GraphNavigationEnvironment:
    """Build a small gold-free graph with candidates inside and outside the pool."""
    nodes = (
        GraphNode(node_id="file:a.py", node_type="file", name="a.py", file_path="a.py"),
        GraphNode(node_id="file:b.py", node_type="file", name="b.py", file_path="b.py"),
        GraphNode(node_id="file:c.py", node_type="file", name="c.py", file_path="c.py"),
        GraphNode(node_id="file:d.py", node_type="file", name="d.py", file_path="d.py"),
        GraphNode(
            node_id="function:a.py:a.inside:1",
            node_type="function",
            name="inside",
            file_path="a.py",
            qualified_name="a.inside",
            start_line=1,
            end_line=2,
        ),
        GraphNode(
            node_id="function:c.py:c.discovered:3",
            node_type="function",
            name="discovered",
            file_path="c.py",
            qualified_name="c.discovered",
            start_line=3,
            end_line=4,
        ),
    )
    files = (
        RankedCandidate(
            rank=1, score=2.0, document_id="file:a.py", kind="file", file_path="a.py"
        ),
        RankedCandidate(
            rank=2, score=1.0, document_id="file:b.py", kind="file", file_path="b.py"
        ),
    )
    functions = (
        RankedCandidate(
            rank=1,
            score=2.0,
            document_id="function:a.py:a.inside:1",
            kind="function",
            file_path="a.py",
            symbol_name="inside",
            qualified_name="a.inside",
            start_line=1,
            end_line=2,
        ),
    )
    return GraphNavigationEnvironment(
        RepositoryGraph(nodes=nodes, edges=()),
        files,
        functions,
        expand_candidate_pool=expanded,
        max_expanded_candidates=cap,
    )


def test_expanded_pool_admits_observed_file_and_function_without_gold() -> None:
    environment = _expansion_environment(expanded=True)
    environment.step(AgentAction(ActionName.INSPECT_FILE, "file:c.py"))
    environment.step(
        AgentAction(ActionName.INSPECT_FUNCTION, "function:c.py:c.discovered:3")
    )

    files, functions = environment.reranked_candidates()

    assert files[0].document_id == "file:c.py"
    assert functions[0].document_id == "function:c.py:c.discovered:3"
    assert len(files) == 2
    assert len(functions) == 1
    assert "gold" not in GraphNavigationEnvironment.__init__.__annotations__


def test_expanded_pool_cap_deduplication_and_order_are_deterministic() -> None:
    first = _expansion_environment(expanded=True, cap=1)
    second = _expansion_environment(expanded=True, cap=1)
    for environment, node_ids in (
        (first, ("file:d.py", "file:c.py", "function:c.py:c.discovered:3")),
        (second, ("function:c.py:c.discovered:3", "file:c.py", "file:d.py")),
    ):
        for node_id in node_ids:
            action = (
                ActionName.INSPECT_FUNCTION
                if node_id.startswith("function:")
                else ActionName.INSPECT_FILE
            )
            environment.step(AgentAction(action, node_id))

    first_files, first_functions = first.reranked_candidates()
    second_files, second_functions = second.reranked_candidates()

    assert [item.document_id for item in first_files] == [
        item.document_id for item in second_files
    ] == ["file:c.py", "file:a.py"]
    assert [item.document_id for item in first_functions] == [
        item.document_id for item in second_functions
    ] == ["function:c.py:c.discovered:3"]


def test_fixed_pool_preserves_original_candidate_membership() -> None:
    environment = _expansion_environment(expanded=False)
    environment.step(AgentAction(ActionName.INSPECT_FILE, "file:c.py"))
    environment.step(
        AgentAction(ActionName.INSPECT_FUNCTION, "function:c.py:c.discovered:3")
    )

    files, functions = environment.reranked_candidates()

    assert [item.document_id for item in files] == ["file:a.py", "file:b.py"]
    assert [item.document_id for item in functions] == ["function:a.py:a.inside:1"]
