"""
Purpose:
    Verify retrieval evaluation behavior without requiring network access.

Input:
    Small temporary repositories, records, models, or fixtures created by the tests.

Output:
    Passing assertions when the implementation follows the Review-1 contract.

Pipeline stage:
    Regression testing for the GraphFix implementation.
"""

from graphfix.evaluation import (
    GoldLocations,
    aggregate_metrics,
    evaluate_rankings,
    normalize_function_target,
    normalize_repository_path,
)
from graphfix.retrieval import BM25Index, RankedCandidate, SearchDocument


def test_bm25_ranks_matching_file_and_function_documents() -> None:
    """Verify bm25 ranks matching file and function documents."""
    documents = (
        SearchDocument(
            document_id="file:cache.py",
            kind="file",
            text="cache expiration timeout eviction",
            file_path="pkg/cache.py",
        ),
        SearchDocument(
            document_id="file:parser.py",
            kind="file",
            text="python syntax parser ast symbols",
            file_path="pkg/parser.py",
        ),
    )
    index = BM25Index(documents)

    ranking = index.search("cache timeout bug", top_k=2)

    assert ranking[0].file_path == "pkg/cache.py"
    assert ranking[0].score > ranking[1].score
    assert [item.rank for item in ranking] == [1, 2]


def test_path_and_function_normalization() -> None:
    """Verify path and function normalization."""
    assert normalize_repository_path(r".\Pkg\Module.py") == "pkg/module.py"
    assert normalize_repository_path("b/pkg/module.py") == "pkg/module.py"
    assert normalize_function_target("b/pkg/module.py:Service.run()") == (
        "pkg/module.py:service.run"
    )


def candidate(rank: int, path: str, name: str | None = None) -> RankedCandidate:
    """Perform candidate."""
    return RankedCandidate(
        rank=rank,
        score=float(10 - rank),
        document_id=f"candidate:{rank}",
        kind="function" if name else "file",
        file_path=path,
        symbol_name=name,
        qualified_name=f"pkg.module.Service.{name}" if name else None,
    )


def test_recall_mrr_and_multi_gold_function_matching() -> None:
    """Verify recall mrr and multi gold function matching."""
    gold = GoldLocations(
        files=("pkg/a.py", "pkg/b.py"),
        functions=("pkg/a.py:target", "pkg/b.py:service.run"),
    )
    metrics = evaluate_rankings(
        (candidate(1, "pkg/x.py"), candidate(2, "pkg/a.py"), candidate(3, "pkg/b.py")),
        (
            candidate(1, "pkg/x.py", "other"),
            candidate(2, "pkg/b.py", "run"),
            candidate(3, "pkg/a.py", "target"),
        ),
        gold,
    )

    assert metrics.file.hit_at_1 == 0.0
    assert metrics.file.recall_at_5 == 1.0
    assert metrics.file.mrr == 0.5
    assert metrics.function.recall_at_5 == 1.0
    assert metrics.function.mrr == 0.5
    aggregate = aggregate_metrics([metrics])
    assert aggregate.file_recall_at_5 == 1.0
    assert aggregate.function_mrr == 0.5


def test_same_function_name_in_two_files_counts_as_two_locations() -> None:
    """Verify same function name in two files counts as two locations."""
    gold = GoldLocations(
        files=("pkg/a.py", "pkg/b.py"),
        functions=("pkg/a.py:thing.run", "pkg/b.py:thing.run"),
    )
    metrics = evaluate_rankings(
        (),
        (RankedCandidate(
            rank=1,
            score=1.0,
            document_id="function:pkg/a.py:Thing.run",
            kind="method",
            file_path="pkg/a.py",
            symbol_name="run",
            qualified_name="pkg.a.Thing.run",
        ),),
        gold,
    )

    assert metrics.function.recall_at_5 == 0.5
