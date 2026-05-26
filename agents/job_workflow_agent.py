from agents.career_agent import analyze_jd_text, analyze_resume_text, match_resume_to_jd
from rag.rag_chain import format_citations, retrieve_sources


def run_job_workflow(
    resume_text: str,
    jd_text: str,
    top_k: int = 3,
) -> dict:
    jd_analysis = analyze_jd_text(jd_text)
    resume_analysis = analyze_resume_text(resume_text)
    match_result = match_resume_to_jd(resume_text, jd_text)

    focus_topics = select_focus_topics(
        jd_keywords=match_result["jd_keywords"],
        missing_keywords=match_result["missing_keywords"],
    )
    rag_query = build_rag_query(focus_topics)
    rag_result = safe_retrieve_sources(query=rag_query, top_k=top_k)

    return {
        "goal": "根据岗位 JD 和候选人简历，自动生成求职准备方案。",
        "workflow_steps": build_workflow_steps(),
        "jd_analysis": jd_analysis,
        "resume_analysis": resume_analysis,
        "match": match_result,
        "rag": {
            "query": rag_query,
            **rag_result,
        },
        "learning_tasks": build_learning_tasks(
            matched_keywords=match_result["matched_keywords"],
            missing_keywords=match_result["missing_keywords"],
            focus_topics=focus_topics,
        ),
        "interview_questions": build_interview_questions(
            resume_text=resume_text,
            jd_text=jd_text,
            jd_keywords=match_result["jd_keywords"],
            matched_keywords=match_result["matched_keywords"],
            missing_keywords=match_result["missing_keywords"],
        ),
        "resume_project_bullets": build_resume_project_bullets(match_result),
        "next_actions": build_next_actions(match_result),
    }


def safe_retrieve_sources(query: str, top_k: int) -> dict:
    try:
        sources, retriever = retrieve_sources(query=query, top_k=top_k)
        return {
            "available": True,
            "error": None,
            "retriever": retriever,
            "citations": format_citations(sources),
            "sources": sources,
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "retriever": {
                "type": "unavailable",
                "top_k": top_k,
                "source_count": 0,
            },
            "citations": [],
            "sources": [],
        }


def build_workflow_steps() -> list[dict]:
    return [
        {
            "step": 1,
            "name": "JD 解析",
            "description": "提取岗位要求里的技术关键词和面试准备重点。",
        },
        {
            "step": 2,
            "name": "简历解析",
            "description": "识别候选人简历中已经出现的技能关键词。",
        },
        {
            "step": 3,
            "name": "匹配度计算",
            "description": "对比 JD 和简历，找出已匹配项和缺失项。",
        },
        {
            "step": 4,
            "name": "RAG 补充",
            "description": "围绕岗位重点检索本地知识库，补充可复习材料和引用来源。",
        },
        {
            "step": 5,
            "name": "行动计划生成",
            "description": "生成学习任务、面试问题和简历项目表达建议。",
        },
    ]


def select_focus_topics(jd_keywords: list[str], missing_keywords: list[str]) -> list[str]:
    if missing_keywords:
        return missing_keywords[:3]

    if jd_keywords:
        return jd_keywords[:3]

    return ["AI Agent", "RAG", "FastAPI"]


def build_rag_query(focus_topics: list[str]) -> str:
    return " ".join(focus_topics) + " 面试 项目 经验 原理 优化"


def build_learning_tasks(
    matched_keywords: list[str],
    missing_keywords: list[str],
    focus_topics: list[str],
) -> list[dict]:
    tasks = []

    for keyword in missing_keywords:
        tasks.append({
            "type": "补齐短板",
            "topic": keyword,
            "task": f"做一个最小 Demo，证明你能在 Agent 项目中使用 {keyword}。",
            "deliverable": f"一段 {keyword} 项目描述 + 一次接口或脚本运行截图。",
        })

    for keyword in matched_keywords[:3]:
        tasks.append({
            "type": "强化优势",
            "topic": keyword,
            "task": f"整理 {keyword} 在当前项目里的使用场景、关键代码和可解释亮点。",
            "deliverable": f"一段能讲给面试官听的 {keyword} 项目复盘。",
        })

    if not tasks:
        for topic in focus_topics:
            tasks.append({
                "type": "基础准备",
                "topic": topic,
                "task": f"补充 {topic} 的核心概念、最小 Demo 和项目接入方式。",
                "deliverable": f"{topic} 学习笔记 + Demo。",
            })

    return tasks


def build_interview_questions(
    resume_text: str,
    jd_text: str,
    jd_keywords: list[str],
    matched_keywords: list[str],
    missing_keywords: list[str],
) -> list[str]:
    questions = []

    resume_evidence = find_keyword_evidence(resume_text, matched_keywords)
    jd_evidence = find_keyword_evidence(jd_text, jd_keywords)

    for keyword in matched_keywords[:5]:
        evidence = resume_evidence.get(keyword)
        questions.extend(build_experience_questions(keyword, evidence))

    for keyword in missing_keywords[:4]:
        jd_line = jd_evidence.get(keyword)
        questions.extend(build_gap_questions(keyword, jd_line))

    if not questions:
        questions.extend([
            "请你用 2 分钟介绍一个最能体现 AI 自动化能力的项目：业务问题是什么、你负责哪部分、最后怎么验证效果？",
            "如果让你把一个重复性办公流程做成 AI 工具，你会怎么拆解需求、设计接口、验证输出质量？",
        ])

    return dedupe_keep_order(questions)[:10]


def split_text_units(text: str) -> list[str]:
    separators = ["\n", "。", "；", ";", "，", ","]
    units = [text.strip()]

    for separator in separators:
        next_units = []
        for unit in units:
            next_units.extend(unit.split(separator))
        units = next_units

    cleaned = []
    for unit in units:
        item = unit.strip(" -\t\r\n")
        if 6 <= len(item) <= 160:
            cleaned.append(item)

    return cleaned


def find_keyword_evidence(text: str, keywords: list[str]) -> dict[str, str]:
    units = split_text_units(text)
    evidence = {}

    for keyword in keywords:
        lower_keyword = keyword.lower()
        for unit in units:
            if lower_keyword in unit.lower():
                evidence[keyword] = unit
                break

    return evidence


def build_experience_questions(keyword: str, evidence: str | None) -> list[str]:
    prefix = f"你简历里提到“{evidence}”。" if evidence else f"你简历里体现了 {keyword}。"
    templates = {
        "RAG": [
            f"{prefix} 这个 RAG 链路里文档是怎么切片的？chunk size、overlap、标题层级你是怎么取舍的？",
            f"{prefix} 你怎么判断检索结果是可靠的？有没有做 score 阈值、top_k、引用来源或召回评测？",
        ],
        "Agent": [
            f"{prefix} 你的 Agent 是怎么决定调用哪个工具的？Router、Tool Executor、trace 分别怎么设计？",
            f"{prefix} 如果工具调用失败或模型返回了错误 JSON，你的 Agent 怎么兜底，怎么避免死循环？",
        ],
        "API": [
            f"{prefix} 你这个 API 是怎么设计请求体和响应体的？哪些字段给前端展示，哪些字段留给调试？",
            f"{prefix} 如果接口被频繁调用，你会怎么做鉴权、限流、日志和错误码设计？",
        ],
        "FastAPI": [
            f"{prefix} FastAPI 里你怎么拆分路由、Pydantic 模型和业务逻辑？为什么不把逻辑全写在接口函数里？",
            f"{prefix} 文件上传接口怎么处理大小限制、格式校验和异常返回？",
        ],
        "Redis": [
            f"{prefix} Redis 在你的项目里存了哪些状态？session、messages、trace 的 key 是怎么设计的？",
            f"{prefix} 如果 Redis 挂了，多轮对话和 trace 会受什么影响？你会怎么降级？",
        ],
        "Docker": [
            f"{prefix} 你的 Docker Compose 里 API、Redis、Chroma 数据目录是怎么组织的？哪些数据需要持久化？",
            f"{prefix} 如果部署后容器重启，session 和向量索引会不会丢？你怎么验证？",
        ],
        "Embedding": [
            f"{prefix} 你为什么选择这个 embedding 模型？向量维度、中文效果、首次加载耗时怎么处理？",
            f"{prefix} embedding 缓存和 Chroma 持久化分别解决什么问题？",
        ],
        "Chroma": [
            f"{prefix} Chroma 里存了哪些 metadata？返回 sources 和 citations 时怎么保证可追溯？",
            f"{prefix} 知识库更新后，索引重建和旧索引清理是怎么做的？",
        ],
        "工作流": [
            f"{prefix} 这个 workflow 每一步的输入输出是什么？哪一步失败时可以继续降级执行？",
            f"{prefix} 你怎么证明它不是普通脚本，而是可观测、可复用、可扩展的业务编排？",
        ],
        "Workflow": [
            f"{prefix} 这个 workflow 每一步的输入输出是什么？哪一步失败时可以继续降级执行？",
            f"{prefix} 你怎么证明它不是普通脚本，而是可观测、可复用、可扩展的业务编排？",
        ],
        "Python": [
            f"{prefix} 你在 Python 里怎么组织模块边界？Agent、RAG、tools、stores 为什么这样拆？",
            f"{prefix} 哪些逻辑你会写成纯函数，哪些逻辑需要依赖外部服务？这样对测试有什么好处？",
        ],
    }

    return templates.get(keyword, [
        f"{prefix} 请结合代码讲一下 {keyword} 在项目中的具体使用位置、输入输出和异常处理。",
    ])


def build_gap_questions(keyword: str, jd_line: str | None) -> list[str]:
    prefix = f"JD 里提到“{jd_line}”。" if jd_line else f"JD 要求 {keyword}，但简历里还没有明显体现。"
    templates = {
        "RAG": f"{prefix} 如果要补一个 RAG 最小版本，你会先支持哪类文档、怎么切片、怎么验证召回是否正确？",
        "Agent": f"{prefix} 如果要补 Agent 能力，你会先做 Router、Tool Calling 还是 Memory？为什么？",
        "API": f"{prefix} 如果要把它做成可调用 API，你会设计哪些接口、状态码和返回字段？",
        "FastAPI": f"{prefix} 如果用 FastAPI 落地，你会怎么定义 Pydantic 模型、文件上传接口和异常处理？",
        "Redis": f"{prefix} 如果补 Redis，你会把它用在缓存、session 还是任务队列？key 和 TTL 怎么设计？",
        "Docker": f"{prefix} 如果补 Docker 部署，你会怎么写 Dockerfile 和 docker-compose？Redis/向量库数据怎么持久化？",
        "工作流": f"{prefix} 如果补自动化工作流，你会把任务拆成哪些节点？每个节点的输入输出是什么？",
        "Workflow": f"{prefix} 如果补自动化工作流，你会把任务拆成哪些节点？每个节点的输入输出是什么？",
        "Embedding": f"{prefix} 如果补 embedding 检索，你会怎么选择模型、生成向量、做相似度检索和缓存？",
        "Chroma": f"{prefix} 如果补 Chroma，你会怎么设计 collection、metadata、索引重建和查询返回？",
    }

    return [templates.get(
        keyword,
        f"{prefix} 如果要在 1 天内补一个能展示的 {keyword} Demo，你会做哪些最小功能，怎么证明它能工作？",
    )]


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def build_resume_project_bullets(match_result: dict) -> list[str]:
    keywords = match_result["matched_keywords"] or match_result["jd_keywords"]
    keyword_text = "、".join(keywords[:5]) if keywords else "Python、FastAPI、RAG"

    return [
        f"围绕岗位要求中的 {keyword_text} 设计并实现 AI 求职助手 Agent，覆盖 JD 分析、简历匹配、RAG 检索和模拟面试流程。",
        "使用 FastAPI 提供 API 服务，Redis 管理多用户 session 和工具调用 trace，Chroma 持久化 RAG 向量索引。",
        "将 JD 解析、简历解析、匹配度计算、知识库检索和面试准备编排为可运行 workflow，提升项目的业务完整度和可展示性。",
    ]


def build_next_actions(match_result: dict) -> list[str]:
    missing_keywords = match_result["missing_keywords"]

    if missing_keywords:
        return [
            f"优先补齐缺失关键词：{', '.join(missing_keywords)}。",
            "为每个缺失关键词补一个能运行的最小 Demo，再把它接入当前 Agent 项目。",
            "用 workflow 输出结果反向修改简历项目描述，保证简历和 JD 表述对齐。",
        ]

    return [
        "当前简历和 JD 技术关键词匹配较好，下一步重点打磨项目讲述。",
        "准备一版 2 分钟项目介绍，包含业务问题、架构设计、关键技术和优化点。",
        "用 Mock Interview 模块围绕已匹配关键词做连续追问训练。",
    ]
