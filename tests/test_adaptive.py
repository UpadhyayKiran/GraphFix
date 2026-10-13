"""Regression tests for the gold-free Adaptive Navigation baseline."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from graphfix.datasets import GraphFixInput
from graphfix.pipeline import AdaptiveNavigationConfig, AdaptiveNavigationPipeline
from graphfix.repositories import RepositoryManager
from graphfix.retrieval import RankedCandidate


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


class _HybridStub:
    def __init__(
        self,
        files: tuple[RankedCandidate, ...],
        functions: tuple[RankedCandidate, ...],
    ) -> None:
        self.files = files
        self.functions = functions

    def localize(self, issue: GraphFixInput, *, method: str) -> Any:
        assert method == "hybrid"
        assert not hasattr(issue, "gold")
        return SimpleNamespace(file_rankings=self.files, function_rankings=self.functions)


def _responses(values: tuple[str, ...]) -> tuple[Iterator[str], Any]:
    responses = iter(values)

    def generate(prompt: str) -> str:
        assert "RAW_GOLD_SENTINEL" not in prompt
        return next(responses)

    return responses, generate


def test_adaptive_pipeline_is_gold_free_serializable_and_reranks_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.name", "GraphFix Test")
    _git(source, "config", "user.email", "graphfix@example.invalid")
    (source / "a.py").write_text("def target():\n    return 1\n", encoding="utf-8")
    (source / "b.py").write_text("def other():\n    return 2\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "fixture")
    commit = _git(source, "rev-parse", "HEAD")
    issue = GraphFixInput(
        instance_id="owner__repo-1",
        repo="owner/repo",
        base_commit=commit,
        problem_statement="target behavior is incorrect",
        category="Bug Report",
    )
    files = (
        RankedCandidate(
            rank=1, score=2.0, document_id="file:b.py", kind="file", file_path="b.py"
        ),
        RankedCandidate(
            rank=2, score=1.0, document_id="file:a.py", kind="file", file_path="a.py"
        ),
    )
    functions = (
        RankedCandidate(
            rank=1,
            score=2.0,
            document_id="function:b.py:b.other:1",
            kind="function",
            file_path="b.py",
            symbol_name="other",
            qualified_name="b.other",
            start_line=1,
            end_line=2,
        ),
        RankedCandidate(
            rank=2,
            score=1.0,
            document_id="function:a.py:a.target:1",
            kind="function",
            file_path="a.py",
            symbol_name="target",
            qualified_name="a.target",
            start_line=1,
            end_line=2,
        ),
    )
    _, generator = _responses(
        (
            '{"action":"INSPECT_FILE","node_id":"file:a.py"}',
            '{"action":"STOP","node_id":null}',
            '{"action":"INSPECT_FUNCTION","node_id":"function:a.py:a.target:1"}',
            '{"action":"STOP","node_id":null}',
        )
    )
    manager = RepositoryManager(tmp_path / "cache", url_builder=lambda value: str(source))
    pipeline = AdaptiveNavigationPipeline(
        manager,
        generator,
        config=AdaptiveNavigationConfig(model="test-controller"),
        hybrid_pipeline=cast(Any, _HybridStub(files, functions)),
    )

    result = pipeline.localize(issue)

    assert result.method == "adaptive"
    assert result.file_rankings[0].file_path == "a.py"
    assert result.function_rankings[0].symbol_name == "target"
    assert result.file_navigation.stopping_reason == "explicit_stop"
    assert result.function_navigation.stopping_reason == "explicit_stop"
    assert result.file_navigation.trace[0].observed_nodes[0].snippet.startswith("def target")
    serialized = result.model_dump_json()
    assert "gold" not in type(result).model_fields
    assert json.loads(serialized)["method"] == "adaptive"
    assert "test-controller" in serialized


def test_adaptive_pipeline_persists_expanded_candidate_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.name", "GraphFix Test")
    _git(source, "config", "user.email", "graphfix@example.invalid")
    (source / "a.py").write_text("def target():\n    return 1\n", encoding="utf-8")
    (source / "b.py").write_text("def other():\n    return 2\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "fixture")
    commit = _git(source, "rev-parse", "HEAD")
    issue = GraphFixInput(
        instance_id="owner__repo-2",
        repo="owner/repo",
        base_commit=commit,
        problem_statement="target behavior is incorrect",
        category="Bug Report",
    )
    files = (
        RankedCandidate(
            rank=1, score=1.0, document_id="file:b.py", kind="file", file_path="b.py"
        ),
    )
    functions = (
        RankedCandidate(
            rank=1,
            score=1.0,
            document_id="function:b.py:b.other:1",
            kind="function",
            file_path="b.py",
            symbol_name="other",
            qualified_name="b.other",
            start_line=1,
            end_line=2,
        ),
    )
    _, generator = _responses(
        (
            '{"action":"INSPECT_FILE","node_id":"file:a.py"}',
            '{"action":"STOP","node_id":null}',
            '{"action":"INSPECT_FUNCTION","node_id":"function:a.py:a.target:1"}',
            '{"action":"STOP","node_id":null}',
        )
    )
    pipeline = AdaptiveNavigationPipeline(
        RepositoryManager(tmp_path / "cache", url_builder=lambda value: str(source)),
        generator,
        config=AdaptiveNavigationConfig(
            model="test-controller", candidate_pool="expanded"
        ),
        hybrid_pipeline=cast(Any, _HybridStub(files, functions)),
    )

    result = pipeline.localize(issue)

    assert result.file_rankings[0].document_id == "file:a.py"
    assert result.function_rankings[0].document_id == "function:a.py:a.target:1"
    assert result.file_navigation.expanded_candidate_ids == ("file:a.py",)
    assert result.function_navigation.expanded_candidate_ids == (
        "function:a.py:a.target:1",
    )


def test_pilot_persists_localization_before_gold_access(
    monkeypatch: Any, tmp_path: Path
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_retrieval_pilot.py"
    spec = importlib.util.spec_from_file_location("adaptive_pilot_test_module", script)
    assert spec is not None and spec.loader is not None
    run_retrieval_pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_retrieval_pilot)
    events: list[str] = []
    localization = SimpleNamespace(file_rankings=(), function_rankings=())
    instance = SimpleNamespace(gold=object())

    def save(result: Any, path: Path) -> None:
        assert result is localization
        events.append(f"save:{path.name}")

    def extract(gold: object) -> object:
        assert gold is instance.gold
        assert events == ["save:adaptive.json"]
        events.append("gold")
        return object()

    def evaluate(files: object, functions: object, gold: object) -> str:
        events.append("evaluate")
        return "metrics"

    monkeypatch.setattr(run_retrieval_pilot, "save_result", save)
    monkeypatch.setattr(run_retrieval_pilot, "extract_gold_locations", extract)
    monkeypatch.setattr(run_retrieval_pilot, "evaluate_rankings", evaluate)
    result = run_retrieval_pilot._persist_then_evaluate(
        cast(Any, localization), instance, tmp_path / "adaptive.json"
    )
    assert result == "metrics"
    assert events == ["save:adaptive.json", "gold", "evaluate"]
