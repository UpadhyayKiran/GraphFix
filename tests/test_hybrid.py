import pytest

from graphfix.retrieval import RankedCandidate, reciprocal_rank_fusion


def _candidate(document_id: str, rank: int, score: float = 1.0) -> RankedCandidate:
    return RankedCandidate(
        rank=rank,
        score=score,
        document_id=document_id,
        kind="file",
        file_path=f"{document_id}.py",
    )


def test_rrf_uses_exact_formula_and_includes_single_ranking_candidates() -> None:
    sparse = (_candidate("shared", 1), _candidate("sparse-only", 2))
    dense = (_candidate("dense-only", 1), _candidate("shared", 2))

    fused = reciprocal_rank_fusion((sparse, dense), top_k=3, rrf_k=60)

    assert [item.document_id for item in fused] == ["shared", "dense-only", "sparse-only"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1].score == pytest.approx(1 / 61)


def test_rrf_breaks_equal_scores_by_document_id_and_reassigns_ranks() -> None:
    fused = reciprocal_rank_fusion(
        ((_candidate("b", 1),), (_candidate("a", 1),)), top_k=2
    )
    assert [(item.document_id, item.rank) for item in fused] == [("a", 1), ("b", 2)]


def test_rrf_supports_function_candidates_and_validates_configuration() -> None:
    function = _candidate("function:a", 1).model_copy(
        update={"kind": "function", "symbol_name": "a"}
    )
    assert reciprocal_rank_fusion(((function,),), top_k=1)[0].kind == "function"
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion(((function,),), rrf_k=0)


def test_rrf_fuses_beyond_display_depth_before_truncating() -> None:
    shared = _candidate("x-shared-at-eleven", 11)
    sparse = tuple(_candidate(f"sparse-{rank:02d}", rank) for rank in range(1, 11)) + (
        shared,
    )
    dense = tuple(_candidate(f"dense-{rank:02d}", rank) for rank in range(1, 11)) + (
        shared,
    )

    fused = reciprocal_rank_fusion((sparse, dense), top_k=10, rrf_k=60)

    assert len(fused) == 10
    assert fused[0].document_id == "x-shared-at-eleven"
    assert fused[0].score == pytest.approx(2 / 71)
    assert "sparse-10" not in {item.document_id for item in fused}
