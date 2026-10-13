from collections.abc import Sequence
from pathlib import Path

import pytest

from graphfix.retrieval import DenseIndex, EmbeddingCache, SearchDocument, cosine_similarity


class FakeEncoder:
    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self.vectors = vectors

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self.vectors[text] for text in texts]


class EmptyEncoder:
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return []


class CountingEncoder(FakeEncoder):
    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        super().__init__(vectors)
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls.append(tuple(texts))
        return super().encode(texts)


def _document(document_id: str, text: str) -> SearchDocument:
    return SearchDocument(
        document_id=document_id,
        kind="file",
        text=text,
        file_path=f"{document_id}.py",
    )


def test_cosine_similarity() -> None:
    assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)
    assert cosine_similarity((0.0, 0.0), (1.0, 0.0)) == 0.0


def test_dense_ranking_descends_and_breaks_ties_by_document_id() -> None:
    documents = (_document("b", "near"), _document("a", "tie"), _document("c", "far"))
    encoder = FakeEncoder(
        {"issue": (1.0, 0.0), "near": (1.0, 0.0), "tie": (1.0, 0.0), "far": (0.0, 1.0)}
    )

    rankings = DenseIndex(documents, encoder).search("issue", top_k=3)

    assert [item.document_id for item in rankings] == ["a", "b", "c"]
    assert [item.rank for item in rankings] == [1, 2, 3]


def test_dense_index_rejects_invalid_limits_and_encoder_shape() -> None:
    document = _document("a", "text")
    with pytest.raises(ValueError, match="one embedding"):
        DenseIndex((document,), EmptyEncoder())

    index = DenseIndex((document,), FakeEncoder({"text": (1.0,), "issue": (1.0,)}))
    with pytest.raises(ValueError, match="non-negative"):
        index.search("issue", top_k=-1)


def test_embedding_cache_miss_then_hit_preserves_dense_ranking(tmp_path: Path) -> None:
    documents = (_document("a", "near"), _document("b", "far"))
    vectors: dict[str, tuple[float, ...]] = {
        "near": (1.0, 0.0),
        "far": (0.0, 1.0),
        "issue": (1.0, 0.0),
    }
    cold_encoder = CountingEncoder(vectors)
    cache = EmbeddingCache(tmp_path)
    cold_embeddings = cache.load_or_encode(
        documents,
        cold_encoder,
        repository="owner/repo",
        base_commit="a" * 40,
        model_name="model-a",
        document_level="file",
    )
    cold_ranking = DenseIndex(
        documents, cold_encoder, document_embeddings=cold_embeddings
    ).search("issue")
    assert cold_encoder.calls[0] == ("near", "far")
    assert tuple(tmp_path.glob("*.npz"))

    warm_encoder = CountingEncoder(vectors)
    warm_embeddings = EmbeddingCache(tmp_path).load_or_encode(
        documents,
        warm_encoder,
        repository="owner/repo",
        base_commit="a" * 40,
        model_name="model-a",
        document_level="file",
    )
    warm_ranking = DenseIndex(
        documents, warm_encoder, document_embeddings=warm_embeddings
    ).search("issue")
    assert warm_encoder.calls == [("issue",)]
    assert warm_embeddings == cold_embeddings
    assert [item.document_id for item in warm_ranking] == [
        item.document_id for item in cold_ranking
    ]
    assert [item.score for item in warm_ranking] == pytest.approx(
        [item.score for item in cold_ranking]
    )


@pytest.mark.parametrize(
    ("commit", "model", "text"),
    (
        ("b" * 40, "model-a", "near"),
        ("a" * 40, "model-b", "near"),
        ("a" * 40, "model-a", "changed"),
    ),
)
def test_embedding_cache_invalidates_changed_identity(
    tmp_path: Path, commit: str, model: str, text: str
) -> None:
    initial = (_document("a", "near"),)
    EmbeddingCache(tmp_path).load_or_encode(
        initial,
        CountingEncoder({"near": (1.0,)}),
        repository="owner/repo",
        base_commit="a" * 40,
        model_name="model-a",
        document_level="file",
    )
    changed = (_document("a", text),)
    encoder = CountingEncoder({text: (2.0,)})
    EmbeddingCache(tmp_path).load_or_encode(
        changed,
        encoder,
        repository="owner/repo",
        base_commit=commit,
        model_name=model,
        document_level="file",
    )
    assert encoder.calls == [(text,)]


def test_corrupt_embedding_cache_is_rebuilt(tmp_path: Path) -> None:
    documents = (_document("a", "near"),)
    cache = EmbeddingCache(tmp_path)
    cache.load_or_encode(
        documents,
        CountingEncoder({"near": (1.0,)}),
        repository="owner/repo",
        base_commit="a" * 40,
        model_name="model-a",
        document_level="file",
    )
    cache_file = next(tmp_path.glob("*.npz"))
    cache_file.write_bytes(b"not an npz file")
    encoder = CountingEncoder({"near": (2.0,)})
    EmbeddingCache(tmp_path).load_or_encode(
        documents,
        encoder,
        repository="owner/repo",
        base_commit="a" * 40,
        model_name="model-a",
        document_level="file",
    )
    assert encoder.calls == [("near",)]
