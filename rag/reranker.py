import hashlib
import math
import re

from rag.query_planner import extract_terms


def source_key(source: dict) -> str:
    raw = "|".join(
        str(source.get(key, ""))
        for key in ("source_file", "title", "chunk_index", "content")
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def normalize_score(score) -> float:
    if score is None:
        return 0.0
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    return value / (1.0 + abs(value)) if value > 1 else max(value, 0.0)


def text_units(text: str) -> set[str]:
    normalized = str(text or "").lower()
    words = set(re.findall(r"[a-z][a-z0-9_+-]*|[\u4e00-\u9fff]{2,}", normalized))
    chinese = re.sub(r"[^\u4e00-\u9fff]", "", normalized)
    bigrams = {chinese[index:index + 2] for index in range(max(len(chinese) - 1, 0))}
    return words | bigrams


def lexical_score(query: str, source: dict) -> tuple[float, list[str]]:
    query_terms = set(extract_terms(query)) | text_units(query)
    if not query_terms:
        return 0.0, []

    title_units = text_units(source.get("title", ""))
    content_units = text_units(source.get("content", ""))
    matched_title = sorted(query_terms & title_units)
    matched_content = sorted(query_terms & content_units)
    matched = sorted(set(matched_title + matched_content))

    title_score = len(matched_title) * 2.5
    content_score = len(matched_content)
    coverage = len(matched) / max(len(query_terms), 1)
    return title_score + content_score + coverage * 3.0, matched[:12]


def dedupe_sources(sources: list[dict]) -> list[dict]:
    merged = {}

    for source in sources:
        key = source_key(source)
        if key not in merged:
            item = dict(source)
            item["retrievers"] = [source.get("retriever", "unknown")]
            item["query_variants"] = [source.get("query_variant", "")]
            merged[key] = item
            continue

        item = merged[key]
        item["score"] = max(normalize_score(item.get("score")), normalize_score(source.get("score")))
        retriever = source.get("retriever", "unknown")
        if retriever not in item["retrievers"]:
            item["retrievers"].append(retriever)
        query_variant = source.get("query_variant", "")
        if query_variant and query_variant not in item["query_variants"]:
            item["query_variants"].append(query_variant)

    return list(merged.values())


def rerank_sources(query: str, sources: list[dict], top_k: int) -> list[dict]:
    deduped = dedupe_sources(sources)
    scored = []

    for source in deduped:
        lexical, matched_terms = lexical_score(query, source)
        retrieval = normalize_score(source.get("score"))
        retriever_bonus = math.log1p(len(source.get("retrievers", []))) * 0.4
        rerank_score = retrieval * 2.0 + lexical + retriever_bonus
        item = dict(source)
        item["retrieval_score"] = source.get("score")
        item["rerank_score"] = round(rerank_score, 4)
        item["matched_terms"] = matched_terms
        scored.append(item)

    scored.sort(key=lambda item: item["rerank_score"], reverse=True)

    for rank, source in enumerate(scored[:top_k], start=1):
        source["rank"] = rank
        source["citation_id"] = f"S{rank}"

    return scored[:top_k]
