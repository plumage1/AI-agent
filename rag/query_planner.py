import re


STOPWORDS = {
    "什么",
    "怎么",
    "如何",
    "为什么",
    "一下",
    "区别",
    "解决",
    "介绍",
    "说明",
    "the",
    "a",
    "an",
    "is",
    "are",
    "how",
    "what",
    "why",
}

SYNONYMS = {
    "缓存雪崩": ["大量缓存同时失效", "redis 缓存雪崩", "cache avalanche"],
    "缓存穿透": ["查询不存在数据", "redis 缓存穿透", "cache penetration"],
    "持久化": ["RDB AOF", "数据不丢", "redis persistence"],
    "RAG": ["检索增强生成", "知识库问答", "retrieval augmented generation"],
    "Agent": ["智能体", "工具调用", "workflow"],
}


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "")).strip()


def extract_terms(query: str) -> list[str]:
    normalized = normalize_query(query)
    terms = re.findall(r"[A-Za-z][A-Za-z0-9_+-]*|[\u4e00-\u9fff]{2,}", normalized)
    result = []

    for term in terms:
        item = term.strip()
        if not item:
            continue
        if item.lower() in STOPWORDS or item in STOPWORDS:
            continue
        if item not in result:
            result.append(item)

    return result


def build_query_variants(query: str, max_variants: int = 4) -> list[str]:
    normalized = normalize_query(query)
    if not normalized:
        return []

    variants = [normalized]
    terms = extract_terms(normalized)

    if terms:
        variants.append(" ".join(terms))

    expanded_terms = []
    for term in terms:
        expanded_terms.append(term)
        for key, values in SYNONYMS.items():
            if term.lower() == key.lower() or key.lower() in normalized.lower():
                expanded_terms.extend(values)

    if expanded_terms:
        variants.append(" ".join(dict.fromkeys(expanded_terms)))

    for key, values in SYNONYMS.items():
        if key.lower() in normalized.lower():
            variants.append(f"{normalized} {' '.join(values)}")

    deduped = []
    for item in variants:
        compact = normalize_query(item)
        if compact and compact not in deduped:
            deduped.append(compact)

    return deduped[:max_variants]
