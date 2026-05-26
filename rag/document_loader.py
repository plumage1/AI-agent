import base64
import os
from io import BytesIO
from pathlib import Path

from core.llm import get_client, model


TEXT_EXTENSIONS = {".md", ".txt"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS | IMAGE_EXTENSIONS
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()


def make_knowledge_filename(source_filename: str) -> str:
    source_name = Path(source_filename).name.strip()

    if not source_name:
        raise ValueError("source filename cannot be empty")

    stem = Path(source_name).stem.strip()

    if not stem:
        raise ValueError("source filename cannot be empty")

    safe_stem = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in stem
    )

    return f"{safe_stem}.md"


def decode_text_file(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return file_bytes.decode("utf-8", errors="replace")


def extract_pdf_text(file_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("pypdf is required to import PDF files") from e

    reader = PdfReader(BytesIO(file_bytes))
    pages = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            pages.append(f"Page {index}\n{text}")

    return "\n\n".join(pages)


def extract_image_text_with_llm(file_bytes: bytes, mime_type: str) -> str:
    image_base64 = base64.b64encode(file_bytes).decode("utf-8")

    response = get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请识别这张图片中的全部文字。"
                            "如果是招聘 JD，请保留岗位职责、任职要求、加分项等结构。"
                            "只输出识别到的正文，不要解释。"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
        stream=False,
    )

    return response.choices[0].message.content or ""


def extract_image_text_with_local_ocr(file_bytes: bytes) -> str:
    try:
        from PIL import Image
        import pytesseract
    except ImportError as e:
        raise RuntimeError(
            "本地图片 OCR 需要安装 Pillow 和 pytesseract。"
        ) from e

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    image = Image.open(BytesIO(file_bytes))
    try:
        text = pytesseract.image_to_string(image, lang="chi_sim+eng")
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError(
            "未找到 Tesseract OCR。请安装 Tesseract，"
            "并确保它已加入 PATH，或在 .env 中配置 TESSERACT_CMD。"
        ) from e

    return text.strip()


def build_image_extract_error(local_error: Exception | None, llm_error: Exception | None) -> RuntimeError:
    parts = ["图片识别失败。"]

    if local_error is not None:
        parts.append(f"本地 OCR：{local_error}")
    else:
        parts.append("本地 OCR：未返回可用文字。")

    if llm_error is not None:
        parts.append("远程视觉：当前不可用或响应超时。")
    else:
        parts.append("远程视觉：未返回可用文字。")

    parts.append("建议先安装并配置 Tesseract OCR；如果仍想走远程识别，再确认当前模型支持图片输入。")

    return RuntimeError(" ".join(parts))


def extract_image_text(file_bytes: bytes, suffix: str) -> str:
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")

    local_error = None
    llm_error = None

    try:
        local_text = extract_image_text_with_local_ocr(file_bytes)
        if local_text:
            return local_text
    except Exception as e:
        local_error = e

    try:
        llm_text = (extract_image_text_with_llm(file_bytes, mime_type=mime_type) or "").strip()
        if llm_text:
            return llm_text
    except Exception as e:
        llm_error = e

    raise build_image_extract_error(local_error, llm_error) from (local_error or llm_error)


def build_knowledge_markdown(title: str, text: str) -> str:
    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError("document content cannot be empty")

    if "\n## " in cleaned_text or cleaned_text.startswith("## "):
        return cleaned_text

    return f"# {title}\n\n## Document Content\n{cleaned_text}\n"


def load_document_as_markdown(source_filename: str, file_bytes: bytes) -> dict:
    suffix = Path(source_filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("only .md, .txt, .pdf, .png, .jpg, .jpeg and .webp files are supported")

    if suffix == ".pdf":
        text = extract_pdf_text(file_bytes)
    elif suffix in IMAGE_EXTENSIONS:
        text = extract_image_text(file_bytes, suffix=suffix)
    else:
        text = decode_text_file(file_bytes)

    knowledge_filename = make_knowledge_filename(source_filename)
    title = Path(knowledge_filename).stem
    content = build_knowledge_markdown(title=title, text=text)

    return {
        "source_filename": Path(source_filename).name,
        "knowledge_filename": knowledge_filename,
        "content": content,
        "char_count": len(content),
    }


def extract_document_text(source_filename: str, file_bytes: bytes) -> str:
    suffix = Path(source_filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("only .md, .txt, .pdf, .png, .jpg, .jpeg and .webp files are supported")

    if suffix == ".pdf":
        text = extract_pdf_text(file_bytes)
    elif suffix in IMAGE_EXTENSIONS:
        text = extract_image_text(file_bytes, suffix=suffix)
    else:
        text = decode_text_file(file_bytes)

    text = text.strip()

    if not text:
        raise ValueError("document content cannot be empty")

    return text
