from __future__ import annotations

from agents.career_agent import extract_tech_keywords


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    result = []

    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)

    return result


def build_structured_learning_plan(topic: str, current_gap: list[str] | None = None) -> dict:
    focus = dedupe_keep_order((current_gap or [])[:3]) or [topic]
    plan_days = [
        {"day": 1, "focus": f"梳理 {topic} 核心概念", "deliverable": "一页知识卡片"},
        {"day": 2, "focus": f"完成 {topic} 最小 Demo", "deliverable": "一段可运行代码或脚本"},
        {"day": 3, "focus": f"把 {topic} 接入当前 Agent 项目", "deliverable": "一段项目接入说明"},
        {"day": 4, "focus": f"围绕 {topic} 补面试表达", "deliverable": "3 个 STAR 口径回答"},
    ]

    return {
        "topic": topic,
        "focus_topics": focus,
        "plan_days": plan_days,
        "summary": f"建议围绕 {topic} 做概念、Demo、项目接入和面试表达四段式准备。",
        "evidence": [f"当前请求聚焦在 {topic}。"] + [f"需要补齐的方向：{item}" for item in focus],
        "next_actions": [
            f"先完成 {topic} 最小 Demo，再整理成简历项目表达。",
            "把学习产物转成可讲的项目故事，而不只是知识点罗列。",
        ],
    }


def synthesize_resume_bullets(
    resume_analysis: dict,
    jd_analysis: dict,
    match_result: dict,
) -> dict:
    matched = dedupe_keep_order(match_result.get("matched_keywords", []))
    missing = dedupe_keep_order(match_result.get("missing_keywords", []))
    jd_keywords = dedupe_keep_order(jd_analysis.get("keywords", []))
    resume_skills = dedupe_keep_order(resume_analysis.get("skills", []))

    strongest = matched[:3] or resume_skills[:3] or ["AI Agent", "FastAPI", "RAG"]
    target = jd_keywords[:4] or ["Python", "RAG", "Redis"]

    bullets = [
        f"设计并实现 AI 求职助手 Agent，围绕 {', '.join(target)} 构建多轮对话、求职 workflow 与模拟面试链路。",
        f"将 {', '.join(strongest)} 等能力落到可运行系统中，支持工具调用、知识检索与求职场景任务编排。",
        "基于 LangGraph / FastAPI / Redis 或内存回退方案组织线程状态、工具轨迹和可解释执行流程。",
    ]

    keyword_reinforcements = [
        f"在项目描述中明确写出 {keyword} 的使用场景、输入输出和效果。"
        for keyword in missing[:3]
    ]
    risk_terms = dedupe_keep_order(
        ["熟悉", "了解", "参与"]
        + [f"不要只写 {keyword}，要补工程场景。" for keyword in missing[:2]]
    )

    evidence = []
    if matched:
        evidence.append(f"已匹配关键词：{', '.join(matched)}")
    if missing:
        evidence.append(f"待补强关键词：{', '.join(missing)}")

    next_actions = [
        "把每条 bullet 补上业务问题、技术动作和结果证据。",
        "优先在第一条项目经历里覆盖 JD 的高频关键词。",
    ]
    if missing:
        next_actions.append(f"优先补写 {', '.join(missing[:3])} 相关项目细节。")

    return {
        "bullets": bullets,
        "keyword_reinforcements": keyword_reinforcements,
        "risk_terms": risk_terms,
        "summary": "已根据简历与 JD 生成更偏面试展示的项目 bullet 和关键词补强建议。",
        "evidence": evidence or ["当前根据简历技能和 JD 关键词生成了通用项目表达。"],
        "next_actions": next_actions,
    }


def prepare_interview_focus(
    jd_analysis: dict,
    match_result: dict,
    rag_result: dict | None = None,
) -> dict:
    keywords = dedupe_keep_order(jd_analysis.get("keywords", []))
    missing = dedupe_keep_order(match_result.get("missing_keywords", []))
    matched = dedupe_keep_order(match_result.get("matched_keywords", []))

    focus_topics = []
    for keyword in missing[:3]:
        focus_topics.append(f"优先补 {keyword}：项目场景、原理边界、为什么之前没做以及现在怎么补。")
    for keyword in matched[:3]:
        focus_topics.append(f"准备 {keyword}：真实项目中怎么落地、踩过什么坑、怎么优化。")

    follow_up_questions = []
    for keyword in keywords[:4]:
        follow_up_questions.append(f"如果面试官追问 {keyword} 在项目中的输入输出和失败降级，你会怎么回答？")

    answer_angles = [
        "先讲业务目标，再讲技术拆解，最后讲结果与复盘。",
        "每个关键词至少准备一个真实项目场景，不要只背定义。",
        "涉及 RAG、Redis、Agent 时，补充异常处理、可观测性和降级策略。",
    ]

    rag_citations = rag_result.get("citations", []) if isinstance(rag_result, dict) else []
    evidence = [f"JD 关键词：{', '.join(keywords[:5])}"] if keywords else []
    if missing:
        evidence.append(f"能力缺口：{', '.join(missing[:4])}")
    if rag_citations:
        evidence.append(f"已命中 {len(rag_citations)} 条知识库引用，可作为复习材料。")

    return {
        "focus_topics": focus_topics or ["先围绕岗位关键词准备项目场景、原理和优化方案。"],
        "follow_up_questions": follow_up_questions or ["请用一个真实项目说明你如何把 AI Agent 能力做成可运行系统。"],
        "answer_angles": answer_angles,
        "summary": "已整理出面试重点、追问方向和推荐答题角度。",
        "evidence": evidence or ["当前根据岗位与匹配结果生成了通用面试准备重点。"],
        "next_actions": [
            "按“项目背景 -> 技术方案 -> 效果指标 -> 问题复盘”准备回答。",
            "先背自己最强的 2 个关键词，再补 1 到 2 个缺口关键词。",
        ],
    }


def job_search_strategy(
    candidate_profile: dict | None = None,
    job_target: dict | None = None,
    current_gap: list[str] | None = None,
) -> dict:
    profile = candidate_profile or {}
    target = job_target or {}
    gaps = dedupe_keep_order(current_gap or [])

    headline = target.get("target_role") or target.get("summary") or "AI Agent 开发岗位"
    strengths = dedupe_keep_order(profile.get("strengths", []))

    seven_day_plan = [
        {"day": 1, "focus": "梳理项目亮点", "deliverable": "1 分钟和 3 分钟两个版本的项目介绍"},
        {"day": 2, "focus": "补简历关键词", "deliverable": "改完的项目 bullet"},
        {"day": 3, "focus": "补最小 Demo", "deliverable": "1 个针对缺口技能的小实验"},
        {"day": 4, "focus": "面试问答", "deliverable": "5 个高频追问答案"},
        {"day": 5, "focus": "知识库复盘", "deliverable": "RAG / Agent / Redis 复习卡"},
        {"day": 6, "focus": "模拟面试", "deliverable": "1 轮自测反馈"},
        {"day": 7, "focus": "投递材料整理", "deliverable": "最终简历 + 针对岗位的自我介绍"},
    ]

    return {
        "target_role": headline,
        "priority_gaps": gaps[:4],
        "strengths": strengths[:4],
        "seven_day_plan": seven_day_plan,
        "application_strategy": [
            "优先投与你现有项目最贴近的 AI Agent / RAG / 后端工程岗位。",
            "每次投递前，把第一段项目经历改成贴近该 JD 的关键词排序。",
            "面试时先打强项，再主动解释缺口补齐路径。",
        ],
        "summary": f"已输出面向 {headline} 的 7 天冲刺策略。",
        "evidence": [
            f"当前优势：{', '.join(strengths[:4]) or '待补充'}",
            f"优先缺口：{', '.join(gaps[:4]) or '暂无明显缺口'}",
        ],
        "next_actions": [
            "先改简历，再做模拟面试，最后准备针对性投递话术。",
            "把缺口补强产物沉淀成截图、Demo 或 STAR 复盘。",
        ],
    }
