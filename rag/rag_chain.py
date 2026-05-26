import os

from core.llm import call_llm
from prompts.rag_prompt import RAG_PROMPT
from rag.query_planner import build_query_variants
from rag.reranker import rerank_sources
from rag.simple_retriever import retrieve


DEFAULT_MIN_SCORE = 0.5
DEFAULT_RETRIEVER = os.getenv("RAG_RETRIEVER", "hybrid")
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", DEFAULT_MIN_SCORE))
RAG_CONTEXT_MAX_CHARS = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "4200"))


def truncate_text(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def build_context(sources: list[dict], max_chars: int = RAG_CONTEXT_MAX_CHARS) -> str:
    if not sources:
        return ""

    blocks = []
    used_chars = 0

    for source in sources:
        content_budget = max(400, min(1200, max_chars - used_chars - 240))
        if content_budget <= 0:
            break

        content = truncate_text(source["content"], content_budget)
        block = (
            f"[{source.get('citation_id', 'S?')}]\n"
            f"引用排名：{source.get('rank', '-')}\n"
            f"来源文件：{source['source_file']}\n"
            f"标题：{source['title']}\n"
            f"相关分数：{source.get('rerank_score', source.get('score', '-'))}\n"
            f"检索器：{','.join(source.get('retrievers', [source.get('retriever', '-')]))}\n"
            f"内容：\n{content}"
        )
        blocks.append(block)
        used_chars += len(block)

        if used_chars >= max_chars:
            break

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


def annotate_sources(sources: list[dict], retriever: str, query_variant: str) -> list[dict]:
    annotated = []
    for source in sources:
        item = dict(source)
        item["retriever"] = retriever
        item["query_variant"] = query_variant
        annotated.append(item)
    return annotated


def retrieve_with_backend(
    retriever_type: str,
    query: str,
    top_k: int,
    min_score: float,
) -> tuple[list[dict], dict]:
    if retriever_type == "chroma":
        from rag.chroma_store import retrieve_by_chroma

        raw_sources = retrieve_by_chroma(query, top_k=top_k)
        sources = filter_sources_by_score(raw_sources, min_score=min_score)
        return annotate_sources(sources, "chroma", query), {
            "type": "chroma",
            "raw_source_count": len(raw_sources),
            "source_count": len(sources),
            "persisted": True,
        }

    if retriever_type == "embedding":
        from rag.embedding_retriever import MODEL_NAME, retrieve_by_embedding

        raw_sources = retrieve_by_embedding(query, top_k=top_k)
        sources = filter_sources_by_score(raw_sources, min_score=min_score)
        return annotate_sources(sources, "embedding", query), {
            "type": "embedding",
            "raw_source_count": len(raw_sources),
            "source_count": len(sources),
            "model": MODEL_NAME,
            "persisted": False,
        }

    sources = retrieve(query, top_k=top_k, min_score=1)
    return annotate_sources(sources, "keyword", query), {
        "type": "keyword",
        "raw_source_count": len(sources),
        "source_count": len(sources),
        "persisted": False,
    }


def retrieve_sources(
    query: str,
    top_k: int = 2,
    min_score: float = RAG_MIN_SCORE,
) -> tuple[list[dict], dict]:
    retriever_type = DEFAULT_RETRIEVER.lower().strip()
    query_variants = build_query_variants(query)
    candidate_k = max(top_k * 4, 8)
    backend_names = ["keyword"]

    if retriever_type == "hybrid":
        backend_names = ["chroma", "keyword"]
    elif retriever_type in {"chroma", "embedding", "keyword"}:
        backend_names = [retriever_type]

    candidates = []
    backend_runs = []
    warnings = []

    for variant in query_variants or [query]:
        for backend_name in backend_names:
            try:
                sources, info = retrieve_with_backend(
                    retriever_type=backend_name,
                    query=variant,
                    top_k=candidate_k,
                    min_score=min_score,
                )
                candidates.extend(sources)
                backend_runs.append({
                    **info,
                    "query_variant": variant,
                })
            except Exception as exc:
                warnings.append({
                    "backend": backend_name,
                    "query_variant": variant,
                    "error": str(exc),
                })

    ranked = rerank_sources(query=query, sources=candidates, top_k=top_k)

    return ranked, {
        "type": retriever_type,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "min_score": min_score,
        "query_variants": query_variants,
        "backend_runs": backend_runs,
        "warnings": warnings,
        "raw_source_count": len(candidates),
        "source_count": len(ranked),
        "reranker": "lexical_plus_retrieval_score",
        "persisted": any(run.get("persisted") for run in backend_runs),
    }


def format_citations(sources: list[dict]) -> list[dict]:
    citations = []

    for source in sources:
        citations.append({
            "id": source.get("citation_id"),
            "rank": source.get("rank"),
            "source_file": source["source_file"],
            "title": source["title"],
            "score": source.get("rerank_score", source.get("score")),
            "retrieval_score": source.get("retrieval_score"),
            "matched_terms": source.get("matched_terms", []),
            "retrievers": source.get("retrievers", [source.get("retriever")]),
            "chunk_index": source.get("chunk_index"),
            "chunk_id": source.get("chunk_id"),
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

请根据知识库内容回答用户问题。回答中引用事实时使用 [S1] 这样的引用编号；
如果知识库没有支持，不要编造。
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
