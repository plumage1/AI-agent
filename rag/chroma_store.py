from pathlib import Path
import hashlib

from rag.embedding_retriever import MODEL_NAME, get_model
from rag.simple_retriever import BASE_DIR, load_chunks


CHROMA_DIR = BASE_DIR / "data" / "chroma"
COLLECTION_NAME = "myagent_knowledge"
DEFAULT_MIN_SCORE = 0.5


def get_chroma_client():
    try:
        import chromadb
    except ImportError as e:
        raise RuntimeError("chromadb is required to use Chroma retriever") from e

    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_chroma_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def make_chunk_id(chunk: dict) -> str:
    raw = f"{chunk['source_file']}|{chunk['title']}|{chunk.get('chunk_index', 1)}|{chunk['content']}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return f"{chunk['source_file']}:{chunk.get('chunk_index', 1)}:{digest}"


def reset_chroma_index() -> None:
    client = get_chroma_client()

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass


def rebuild_chroma_index() -> dict:
    reset_chroma_index()

    chunks = load_chunks()
    collection = get_chroma_collection()

    if not chunks:
        return {
            "collection": COLLECTION_NAME,
            "persist_directory": str(CHROMA_DIR),
            "chunk_count": 0,
            "model": MODEL_NAME,
        }

    ids = [make_chunk_id(chunk) for chunk in chunks]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [
        {
            "source_file": chunk["source_file"],
            "title": chunk["title"],
            "chunk_index": chunk.get("chunk_index", 1),
        }
        for chunk in chunks
    ]
    embeddings = get_model().encode(documents).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return {
        "collection": COLLECTION_NAME,
        "persist_directory": str(CHROMA_DIR),
        "chunk_count": len(chunks),
        "model": MODEL_NAME,
    }


def get_chroma_status() -> dict:
    collection = get_chroma_collection()

    return {
        "collection": COLLECTION_NAME,
        "persist_directory": str(CHROMA_DIR),
        "chunk_count": collection.count(),
        "model": MODEL_NAME,
    }


def retrieve_by_chroma(query: str, top_k: int = 2) -> list[dict]:
    collection = get_chroma_collection()
    count = collection.count()

    if count == 0:
        rebuild_chroma_index()
        collection = get_chroma_collection()
        count = collection.count()

    if count == 0:
        return []

    query_embedding = get_model().encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    sources = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        score = 1 / (1 + distance)

        sources.append({
            "rank": rank,
            "source_file": metadata["source_file"],
            "title": metadata["title"],
            "content": document,
            "chunk_index": metadata.get("chunk_index", 1),
            "distance": distance,
            "score": score,
        })

    return sources
