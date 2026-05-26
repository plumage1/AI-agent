import math

from rag.simple_retriever import load_chunks


MODEL_NAME = "BAAI/bge-small-zh-v1.5"

_model = None
_cached_chunks = None
_cached_embeddings = None


def get_model():
    global _model

    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is required to use embedding retrieval"
            ) from e

        _model = SentenceTransformer(MODEL_NAME)

    return _model


def cosine_similarity(a, b) -> float:
    if len(a) != len(b):
        return 0.0

    dot_product = sum(left * right for left, right in zip(a, b))
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    denominator = norm_a * norm_b

    if denominator == 0:
        return 0.0

    return float(dot_product / denominator)


def build_embeddings():
    global _cached_chunks, _cached_embeddings

    if _cached_chunks is not None and _cached_embeddings is not None:
        return _cached_chunks, _cached_embeddings

    chunks = load_chunks()
    texts = [chunk["content"] for chunk in chunks]

    embeddings = get_model().encode(texts)

    _cached_chunks = chunks
    _cached_embeddings = embeddings

    return _cached_chunks, _cached_embeddings


def clear_embedding_cache() -> None:
    global _cached_chunks, _cached_embeddings

    _cached_chunks = None
    _cached_embeddings = None


def get_embedding_cache_status() -> dict:
    return {
        "cached": _cached_chunks is not None and _cached_embeddings is not None,
        "chunk_count": len(_cached_chunks) if _cached_chunks is not None else 0
    }


def retrieve_by_embedding(query: str, top_k: int = 2) -> list[dict]:
    chunks, embeddings = build_embeddings()

    if not chunks:
        return []

    query_embedding = get_model().encode(query)

    scored = []

    for chunk, embedding in zip(chunks, embeddings):
        score = cosine_similarity(query_embedding, embedding)

        scored.append({
            **chunk,
            "score": score
        })

    scored.sort(key=lambda item: item["score"], reverse=True)

    return scored[:top_k]
