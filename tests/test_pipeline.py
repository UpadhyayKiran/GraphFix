"""
Purpose:
    Verify pipeline behavior without requiring network access.
Input:
    Small temporary repositories, records, models, or fixtures created by the tests.

Output:
    Passing assertions when the implementation follows the Review-1 contract.

Pipeline stage:
    Regression testing for the GraphFix implementation.
"""

import subprocess
from pathlib import Path

from graphfix.datasets.models import (
    GraphFixGold,
    GraphFixInput,
    GraphFixInstance,
    GraphFixMetadata,
)
from graphfix.pipeline import BM25LocalizationPipeline, Review1Pipeline, save_result
from graphfix.pipeline.campaign import _load_evaluation
from graphfix.pipeline.models import RetrievalConfig
from graphfix.repositories import RepositoryManager


def git(repository: Path, *args: str) -> str:
    """Perform git."""
    result = subprocess.run(
        ["git", *args], cwd=repository, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def test_end_to_end_pipeline_keeps_gold_out_of_localization(tmp_path: Path) -> None:
    """Verify end to end pipeline keeps gold out of localization."""
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-b", "main")
    git(source, "config", "user.name", "GraphFix Test")
    git(source, "config", "user.email", "graphfix@example.invalid")
    (source / "cache.py").write_text(
        "def expire_cache(timeout: int) -> None:\n"
        "    \"\"\"Remove stale cached records after timeout.\"\"\"\n"
        "    return None\n",
        encoding="utf-8",
    )
    (source / "parser.py").write_text("def parse_source():\n    return None\n", encoding="utf-8")
    git(source, "add", ".")
    git(source, "commit", "-m", "fixture")
    commit = git(source, "rev-parse", "HEAD")
    instance = GraphFixInstance(
        input=GraphFixInput(
            instance_id="owner__repo-1",
            repo="owner/repo",
            base_commit=commit,
            problem_statement="Cache timeout does not expire stale records",
            category="Bug Report",
        ),
        gold=GraphFixGold(
            patch=(
                "diff --git a/cache.py b/cache.py\n"
                "--- a/cache.py\n+++ b/cache.py\n"
                "RAW_PATCH_SENTINEL"
            ),
            test_patch="RAW_TEST_PATCH_SENTINEL",
            edit_functions=["cache.py:expire_cache"],
            added_functions=[],
            labels=None,
        ),
        metadata=GraphFixMetadata(hints_text="", created_at=1, edit_functions_length=1),
    )
    manager = RepositoryManager(
        tmp_path / "cache",
        url_builder=lambda _repository: str(source),
    )
    pipeline = BM25LocalizationPipeline(manager)

    localization = pipeline.localize(instance.input)

    serialized = localization.model_dump_json()
    assert localization.checked_out_commit == commit
    assert localization.file_rankings[0].file_path == "cache.py"
    assert localization.function_rankings[0].symbol_name == "expire_cache"
    assert "RAW_PATCH_SENTINEL" not in serialized
    assert "RAW_TEST_PATCH_SENTINEL" not in serialized
    assert "gold" not in type(localization).model_fields

    evaluation = pipeline.evaluate(instance)
    assert evaluation.metrics.file.hit_at_1 == 1.0
    assert evaluation.metrics.function.hit_at_1 == 1.0

    output = tmp_path / "evaluation.json"
    save_result(evaluation, output)
    assert _load_evaluation(output, RetrievalConfig()) is not None
    assert _load_evaluation(output, RetrievalConfig(top_k=20)) is None


def test_review1_pipeline_compatibility_alias() -> None:
    assert Review1Pipeline is BM25LocalizationPipeline
