import os

from core.llm import call_llm
from prompts.rag_prompt import RAG_PROMPT
from rag.chroma_store import DEFAULT_MIN_SCORE
from rag.simple_retriever import retrieve


DEFAULT_RETRIEVER = os.getenv("RAG_RETRIEVER", "chroma")
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", DEFAULT_MIN_SCORE))


def build_context(sources: list[dict]) -> str:
    if not sources:
        return ""

    blocks = []

    for source in sources:
        blocks.append(
            f"引用排名：{source.get('rank', '-')}\n"
            f"来源文件：{source['source_file']}\n"
            f"标题：{source['title']}\n"
            f"相关分数：{source.get('score', '-')}\n"
            f"内容：\n{source['content']}"
        )

    return "\n\n---\n\n".join(blocks)


def filter_sources_by_score(
    sources: list[dict],
    min_score: float = RAG_MIN_SCORE,
) -> list[dict]:
    filtered = []

    for source in sources:
        score = source.get("score")

        if score is None or score >= min_score:
            filtered.append(source)

    return filtered


def retrieve_sources(
    query: str,
    top_k: int = 2,
    min_score: float = RAG_MIN_SCORE,
) -> tuple[list[dict], dict]:
    retriever_type = DEFAULT_RETRIEVER.lower().strip()

    if retriever_type == "chroma":
        from rag.chroma_store import retrieve_by_chroma

        raw_sources = retrieve_by_chroma(query, top_k=top_k)
        sources = filter_sources_by_score(raw_sources, min_score=min_score)
        retriever_info = {
            "type": "chroma",
            "top_k": top_k,
            "min_score": min_score,
            "raw_source_count": len(raw_sources),
            "source_count": len(sources),
            "persisted": True,
        }
        return sources, retriever_info

    if retriever_type == "embedding":
        from rag.embedding_retriever import MODEL_NAME, retrieve_by_embedding

        raw_sources = retrieve_by_embedding(query, top_k=top_k)
        sources = filter_sources_by_score(raw_sources, min_score=min_score)
        retriever_info = {
            "type": "embedding",
            "top_k": top_k,
            "min_score": min_score,
            "raw_source_count": len(raw_sources),
            "source_count": len(sources),
            "model": MODEL_NAME,
            "persisted": False,
        }
        return sources, retriever_info

    sources = retrieve(query, top_k=top_k, min_score=10)
    retriever_info = {
        "type": "keyword",
        "top_k": top_k,
        "min_score": 10,
        "raw_source_count": len(sources),
        "source_count": len(sources),
        "persisted": False,
    }
    return sources, retriever_info


def format_citations(sources: list[dict]) -> list[dict]:
    citations = []

    for source in sources:
        citations.append({
            "rank": source.get("rank"),
            "source_file": source["source_file"],
            "title": source["title"],
            "score": source.get("score"),
            "chunk_index": source.get("chunk_index"),
        })

    return citations


def answer_with_rag(query: str) -> str:
    result = answer_with_rag_and_sources(query)
    return result["answer"]


def answer_with_rag_and_sources(
    query: str,
    top_k: int = 2,
    min_score: float = RAG_MIN_SCORE,
) -> dict:
    sources, retriever_info = retrieve_sources(
        query=query,
        top_k=top_k,
        min_score=min_score,
    )
    context = build_context(sources)
    citations = format_citations(sources)

    if not context:
        return {
            "answer": "知识库中没有足够相关的信息。",
            "sources": [],
            "citations": [],
            "retriever": retriever_info,
        }

    messages = [
        {"role": "system", "content": RAG_PROMPT},
        {
            "role": "user",
            "content": f"""
用户问题：
{query}

知识库内容：
{context}

请根据知识库内容回答用户问题。回答末尾请简要列出参考来源标题。
""",
        },
    ]

    answer = call_llm(messages)

    return {
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "retriever": retriever_info,
    }
