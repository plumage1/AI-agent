from rag.document_loader import extract_document_text


TECH_KEYWORDS = [
    "Python",
    "FastAPI",
    "Redis",
    "RAG",
    "Agent",
    "Docker",
    "Chroma",
    "ChromaDB",
    "向量数据库",
    "Embedding",
    "Tool Calling",
    "LangGraph",
    "LangChain",
    "SQL",
    "MySQL",
    "PostgreSQL",
    "Java",
    "Linux",
    "Nginx",
    "API",
    "异步",
    "缓存",
    "消息队列",
    "部署",
    "工作流",
    "Workflow",
]


def extract_tech_keywords(text: str) -> list[str]:
    matched = []
    lower_text = text.lower()

    for keyword in TECH_KEYWORDS:
        if keyword.lower() in lower_text and keyword not in matched:
            matched.append(keyword)

    return matched


def analyze_resume_text(resume_text: str) -> dict:
    skills = extract_tech_keywords(resume_text)

    return {
        "skills": skills,
        "skill_count": len(skills),
        "suggestions": build_resume_suggestions(skills),
    }


def analyze_jd_text(jd_text: str) -> dict:
    keywords = extract_tech_keywords(jd_text)

    return {
        "keywords": keywords,
        "keyword_count": len(keywords),
        "preparation_focus": build_interview_focus(keywords),
    }


def match_resume_to_jd(resume_text: str, jd_text: str) -> dict:
    resume_keywords = extract_tech_keywords(resume_text)
    jd_keywords = extract_tech_keywords(jd_text)
    matched_keywords = [
        keyword
        for keyword in jd_keywords
        if keyword in resume_keywords
    ]
    missing_keywords = [
        keyword
        for keyword in jd_keywords
        if keyword not in resume_keywords
    ]

    match_score = calculate_match_score(
        matched_count=len(matched_keywords),
        total_count=len(jd_keywords),
    )

    return {
        "match_score": match_score,
        "resume_keywords": resume_keywords,
        "jd_keywords": jd_keywords,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "resume_suggestions": build_match_suggestions(matched_keywords, missing_keywords),
        "interview_focus": build_interview_focus(jd_keywords),
    }


def analyze_resume_file(filename: str, file_bytes: bytes) -> dict:
    text = extract_document_text(filename, file_bytes)
    result = analyze_resume_text(text)
    result["source_filename"] = filename
    result["char_count"] = len(text)
    return result


def analyze_jd_file(filename: str, file_bytes: bytes) -> dict:
    text = extract_document_text(filename, file_bytes)
    result = analyze_jd_text(text)
    result["source_filename"] = filename
    result["char_count"] = len(text)
    return result


def match_resume_file_to_jd(filename: str, file_bytes: bytes, jd_text: str) -> dict:
    text = extract_document_text(filename, file_bytes)
    result = match_resume_to_jd(text, jd_text)
    result["source_filename"] = filename
    result["resume_char_count"] = len(text)
    return result


def match_resume_file_to_jd_file(
    resume_filename: str,
    resume_file_bytes: bytes,
    jd_filename: str,
    jd_file_bytes: bytes,
) -> dict:
    resume_text = extract_document_text(resume_filename, resume_file_bytes)
    jd_text = extract_document_text(jd_filename, jd_file_bytes)
    result = match_resume_to_jd(resume_text, jd_text)
    result["source_filename"] = resume_filename
    result["resume_char_count"] = len(resume_text)
    result["jd_source_filename"] = jd_filename
    result["jd_char_count"] = len(jd_text)
    return result


def calculate_match_score(matched_count: int, total_count: int) -> int:
    if total_count == 0:
        return 0

    return round(matched_count / total_count * 100)


def build_resume_suggestions(skills: list[str]) -> list[str]:
    if not skills:
        return [
            "简历中没有识别到明确技术关键词，建议补充项目技术栈。",
            "不要只写“熟悉某技术”，要写清楚使用场景、解决的问题和结果。",
        ]

    return [
        "将核心技能和具体项目绑定，说明你在项目中如何使用这些技术。",
        "为每个核心技能补充工程场景，例如接口设计、缓存、RAG 检索、部署或性能优化。",
        "尽量使用可验证结果，例如接口数量、支持的文档类型、召回准确率、响应时间或部署方式。",
    ]


def build_match_suggestions(
    matched_keywords: list[str],
    missing_keywords: list[str],
) -> list[str]:
    suggestions = []

    if matched_keywords:
        suggestions.append(
            f"保留并强化这些已匹配技术：{', '.join(matched_keywords)}。"
        )

    if missing_keywords:
        suggestions.append(
            f"补充这些 JD 关键词对应的项目描述：{', '.join(missing_keywords)}。"
        )
        suggestions.append(
            "如果你确实做过相关内容，把关键词写进项目经历；如果还没做过，把它加入后续学习和项目补强计划。"
        )

    if not suggestions:
        suggestions.append("JD 中没有识别到明确技术关键词，建议提供更完整岗位描述。")

    return suggestions


def build_interview_focus(keywords: list[str]) -> list[str]:
    focus = []

    for keyword in keywords:
        focus.append(f"准备 {keyword} 的项目使用场景、核心原理、常见问题和优化方案。")

    if not focus:
        focus.append("先补充完整 JD，再围绕岗位关键词准备面试问题。")

    return focus
