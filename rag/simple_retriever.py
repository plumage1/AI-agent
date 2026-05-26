from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 120


def split_text_by_size(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    cleaned_text = text.strip()

    if not cleaned_text:
        return []

    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    chunks = []
    start = 0

    while start < len(cleaned_text):
        end = min(start + chunk_size, len(cleaned_text))
        window = cleaned_text[start:end]

        if end < len(cleaned_text):
            break_points = [
                window.rfind("\n\n"),
                window.rfind("\n"),
                window.rfind("。"),
                window.rfind("."),
            ]
            break_at = max(break_points)

            if break_at > chunk_size * 0.5:
                end = start + break_at + 1
                window = cleaned_text[start:end]

        chunk = window.strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned_text):
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def split_chunks(text: str, source_file: str) -> list[dict]:
    chunks = []

    for block in text.split("## "):
        block = block.strip()
        if not block:
            continue

        if block.startswith("#"):
            continue

        title, _, body = block.partition("\n")
        title = title.strip()
        body = body.strip()

        for index, chunk_text in enumerate(split_text_by_size(body), start=1):
            chunk_title = title

            if len(body) > DEFAULT_CHUNK_SIZE:
                chunk_title = f"{title} - Part {index}"

            chunks.append({
                "source_file": source_file,
                "title": chunk_title,
                "content": f"{title}\n{chunk_text}",
                "chunk_index": index,
            })

    return chunks


def load_chunks() -> list[dict]:
    chunks = []

    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8-sig")
        chunks.extend(split_chunks(text, source_file=path.name))

    return chunks


def extract_keywords(query: str) -> list[str]:
    keywords = []

    known_keywords = [
        "Redis",
        "缓存雪崩",
        "缓存穿透",
        "持久化",
        "RDB",
        "AOF",
        "RAG",
        "检索增强生成",
        "微调",
        "知识库",
    ]

    for keyword in known_keywords:
        if keyword.lower() in query.lower():
            keywords.append(keyword)

    return keywords


def keyword_score(keyword: str) -> int:
    generic_keywords = {"Redis", "RAG", "知识库"}

    if keyword in generic_keywords:
        return 1

    return 10


def retrieve(query: str, top_k: int = 2, min_score: int = 10) -> list[dict]:
    chunks = load_chunks()
    keywords = extract_keywords(query)

    if not keywords:
        keywords = query.split()

    scored_chunks = []

    for chunk in chunks:
        score = 0
        content = chunk["content"]

        for keyword in keywords:
            if keyword.lower() in content.lower():
                score += keyword_score(keyword)

        source = {
            **chunk,
            "score": score,
        }
        scored_chunks.append(source)

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    return [
        source
        for source in scored_chunks[:top_k]
        if source["score"] >= min_score
    ]
