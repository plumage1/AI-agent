from datetime import datetime
from pathlib import Path

from rag.simple_retriever import KNOWLEDGE_DIR


INVALID_FILENAME_CHARS = set('<>:"/\\|?*')


def normalize_knowledge_filename(filename: str) -> str:
    name = filename.strip()

    if not name:
        raise ValueError("filename cannot be empty")

    if Path(name).name != name:
        raise ValueError("filename cannot contain path separators")

    if any(char in INVALID_FILENAME_CHARS for char in name):
        raise ValueError("filename contains invalid characters")

    if not name.endswith(".md"):
        raise ValueError("knowledge document must be a .md file")

    return name


def get_knowledge_path(filename: str) -> Path:
    name = normalize_knowledge_filename(filename)
    return KNOWLEDGE_DIR / name


def list_knowledge_documents() -> list[dict]:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    documents = []

    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        stat = path.stat()
        documents.append({
            "filename": path.name,
            "size": stat.st_size,
            "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        })

    return documents


def read_knowledge_document(filename: str) -> dict:
    path = get_knowledge_path(filename)

    if not path.exists():
        raise FileNotFoundError(filename)

    return {
        "filename": path.name,
        "content": path.read_text(encoding="utf-8-sig"),
    }


def save_knowledge_document(filename: str, content: str) -> dict:
    path = get_knowledge_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return read_knowledge_document(path.name)


def delete_knowledge_document(filename: str) -> bool:
    path = get_knowledge_path(filename)

    if not path.exists():
        return False

    path.unlink()
    return True
