import hashlib
import re
from pathlib import Path

from rag.query_planner import extract_terms


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


def make_chunk_id(source_file: str, title: str, index: int, content: str) -> str:
    raw = f"{source_file}|{title}|{index}|{content}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def split_chunks(text: str, source_file: str) -> list[dict]:
    chunks = []

    for block in text.split("## "):
        block = block.strip()
        if not block or block.startswith("#"):
            continue

        title, _, body = block.partition("\n")
        title = title.strip()
        body = body.strip()

        for index, chunk_text in enumerate(split_text_by_size(body), start=1):
            chunk_title = title

            if len(body) > DEFAULT_CHUNK_SIZE:
                chunk_title = f"{title} - Part {index}"

            content = f"{title}\n{chunk_text}"
            chunks.append({
                "source_file": source_file,
                "title": chunk_title,
                "content": content,
                "chunk_index": index,
                "chunk_id": make_chunk_id(source_file, chunk_title, index, content),
            })

    return chunks


def load_chunks() -> list[dict]:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    chunks = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8-sig")
        chunks.extend(split_chunks(text, source_file=path.name))
    return chunks


def keyword_score(term: str, title: str, content: str) -> int:
    term_lower = term.lower()
    title_lower = title.lower()
    content_lower = content.lower()
    score = 0

    if term_lower in title_lower:
        score += 5
    if term_lower in content_lower:
        score += 2

    if re.fullmatch(r"[\u4e00-\u9fff]{2,}", term):
        for index in range(len(term) - 1):
            bigram = term[index:index + 2]
            if bigram in title:
                score += 2
            if bigram in content:
                score += 1

    return score


def retrieve(query: str, top_k: int = 2, min_score: int = 1) -> list[dict]:
    chunks = load_chunks()
    terms = extract_terms(query)

    if not terms:
        terms = query.split()

    scored_chunks = []

    for chunk in chunks:
        score = 0
        for term in terms:
            score += keyword_score(term, chunk["title"], chunk["content"])

        source = {
            **chunk,
            "score": score,
            "retriever": "keyword",
        }
        scored_chunks.append(source)

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    return [
        source
        for source in scored_chunks[:top_k]
        if source["score"] >= min_score
    ]
