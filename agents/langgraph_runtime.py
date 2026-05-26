import json
import re
from datetime import datetime
from functools import lru_cache

from agents.career_agent import analyze_jd_text, analyze_resume_text, extract_tech_keywords, match_resume_to_jd
from agents.langgraph_state import AgentState
from core.langgraph_checkpoint import get_checkpointer, get_thread_config
from core.llm import call_llm, complete_message, get_provider
from prompts.tool_summary_prompt import TOOL_SUMMARY_PROMPT
from rag.rag_chain import answer_with_rag_and_sources, format_citations, retrieve_sources
from tools.agent_tools import (
    build_structured_learning_plan,
    job_search_strategy,
    prepare_interview_focus,
    synthesize_resume_bullets,
)
from tools.registry import TOOLS, tool_schemas


TOOL_TRIGGER_KEYWORDS = (
    "学习", "路线", "计划", "roadmap",
    "简历", "resume", "cv",
    "jd", "岗位", "职位", "job description",
    "匹配", "match",
    "rag", "知识库", "检索",
    "redis", "缓存", "向量", "embedding", "chroma",
    "面试", "八股", "原理", "怎么回答", "是什么",
)

TECHNICAL_QUERY_KEYWORDS = (
    "redis", "rag", "fastapi", "langgraph", "langchain",
    "embedding", "vector", "chroma", "docker", "agent",
    "缓存", "向量", "知识库", "检索", "面试", "原理", "区别",
)

AGENT_COMPLEXITY_KEYWORDS = (
    "完整", "系统", "一步步", "逐步", "求职准备", "优化方向", "给我计划",
    "帮我根据", "冲刺", "策略", "路线图", "复盘", "准备方案",
)

MAX_AGENT_STEPS = 5
MAX_STEP_RETRIES = 1


def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build_default_memory() -> dict:
    return {
        "candidate_profile": {
            "skills": [],
            "strengths": [],
            "weaknesses": [],
            "project_highlights": [],
        },
        "job_target": {
            "target_role": "",
            "keywords": [],
            "summary": "",
        },
        "artifact_memory": {
            "last_match_score": None,
            "last_learning_topic": "",
            "last_resume_bullets": [],
            "last_interview_focus": [],
        },
        "last_updated_at": "",
    }


def build_default_agent_snapshot() -> dict:
    return {
        "agent_mode": False,
        "goal": "",
        "task_type": "",
        "plan": [],
        "current_step": None,
        "step_history": [],
        "agent_status": "idle",
        "next_action": "",
        "artifacts": {},
        "final_summary": "",
    }


def ensure_memory(memory: dict | None) -> dict:
    merged = build_default_memory()
    incoming = memory or {}

    for section in ("candidate_profile", "job_target", "artifact_memory"):
        merged[section].update(incoming.get(section, {}))

    if incoming.get("last_updated_at"):
        merged["last_updated_at"] = incoming["last_updated_at"]

    return merged


def should_offer_tools(user_input: str) -> bool:
    lowered = (user_input or "").lower()
    return any(keyword in lowered for keyword in TOOL_TRIGGER_KEYWORDS)


def summarize_tool_results(user_input: str, tool_results: list[dict]) -> str:
    messages = [
        {"role": "system", "content": TOOL_SUMMARY_PROMPT},
        {
            "role": "user",
            "content": (
                f"用户问题：{user_input}\n\n"
                f"工具执行结果：\n{json.dumps(tool_results, ensure_ascii=False, indent=2)}"
            ),
        },
    ]
    return call_llm(messages)


def normalize_tool_calls(tool_calls) -> list[dict]:
    if not tool_calls:
        return []

    normalized = []
    for index, call in enumerate(tool_calls):
        if isinstance(call, dict):
            normalized.append(call)
            continue

        function = getattr(call, "function", None)
        normalized.append({
            "id": getattr(call, "id", f"call_{index}"),
            "type": getattr(call, "type", "function"),
            "function": {
                "name": getattr(function, "name", ""),
                "arguments": getattr(function, "arguments", "{}"),
            },
        })
    return normalized


def parse_tool_arguments(arguments) -> dict:
    if isinstance(arguments, dict):
        return arguments

    if not arguments:
        return {}

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def append_trace(
    traces: list,
    tool_name: str,
    arguments: dict,
    tool_result,
    metadata: dict,
    mode: str,
) -> int:
    traces.append({
        "tool_name": tool_name,
        "arguments": arguments,
        "result": tool_result,
        "metadata": metadata,
        "mode": mode,
        "created_at": now_ts(),
    })
    return len(traces) - 1


def short_json(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(value)


def build_result_summary(result) -> str:
    if isinstance(result, str):
        return result.strip()[:220]

    if not isinstance(result, dict):
        return str(result)[:220]

    for key in ("summary", "answer", "feedback"):
        if result.get(key):
            return str(result[key]).strip()[:220]

    for key in ("keywords", "skills", "matched_keywords", "missing_keywords", "bullets", "focus_topics"):
        values = result.get(key)
        if isinstance(values, list) and values:
            return f"{key}: {', '.join(str(item) for item in values[:4])}"

    return short_json(result)[:220]


def execute_tool_with_metadata(tool_name: str, arguments: dict) -> dict:
    metadata = {}

    if tool_name == "rag_search":
        rag_result = answer_with_rag_and_sources(arguments.get("query", ""))
        metadata["sources"] = rag_result["sources"]
        metadata["citations"] = rag_result.get("citations", [])
        metadata["source_count"] = len(rag_result["sources"])
        metadata["retriever"] = rag_result.get("retriever", {})
        return {
            "result": rag_result["answer"],
            "metadata": metadata,
        }

    if tool_name not in TOOLS:
        return {
            "result": f"Tool {tool_name} does not exist.",
            "metadata": metadata,
        }

    try:
        tool_func = TOOLS[tool_name]["function"]
        result = tool_func(**(arguments or {}))
        if isinstance(result, dict):
            metadata["structured"] = True
        return {
            "result": result,
            "metadata": metadata,
        }
    except TypeError as exc:
        return {
            "result": f"Tool {tool_name} received invalid arguments: {exc}",
            "metadata": metadata,
        }


def extract_labeled_block(text: str, labels: tuple[str, ...]) -> str | None:
    for label in labels:
        pattern = rf"(?:^|\n)\s*{re.escape(label)}\s*[:：]\s*(.+?)(?=(?:\n\s*[A-Za-z\u4e00-\u9fff ]+\s*[:：])|$)"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            if content:
                return content
    return None


def guess_learning_topic(user_input: str) -> str:
    lowered = user_input.lower()
    for topic in ("langgraph", "langchain", "rag", "redis", "fastapi", "docker", "agent"):
        if topic in lowered:
            return topic.upper() if topic == "rag" else topic
    return user_input.strip()[:80] or "AI Agent"


def infer_resume_text(user_input: str) -> str | None:
    text = extract_labeled_block(user_input, ("简历", "resume", "cv"))
    if text:
        return text
    lowered = user_input.lower()
    if any(keyword in lowered for keyword in ("简历", "resume", "cv")) and len(user_input) > 120:
        return user_input
    return None


def infer_jd_text(user_input: str) -> str | None:
    text = extract_labeled_block(user_input, ("jd", "岗位", "职位", "job description"))
    if text:
        return text
    lowered = user_input.lower()
    if any(keyword in lowered for keyword in ("jd", "岗位", "职位", "job description")) and len(user_input) > 120:
        return user_input
    return None


def infer_target_role(user_input: str, memory: dict) -> str:
    lowered = user_input.lower()
    if "后端" in user_input or "backend" in lowered:
        return "AI Agent / 后端开发岗位"
    if "rag" in lowered:
        return "RAG / Agent 开发岗位"
    remembered = memory.get("job_target", {}).get("target_role")
    return remembered or "AI Agent 开发岗位"


def collect_context_inputs(user_input: str, state: AgentState) -> dict:
    memory = ensure_memory(state.get("memory"))
    artifacts = state.get("artifacts", {}) or {}

    topic = guess_learning_topic(user_input)
    if "面试" in user_input:
        topic = extract_labeled_block(user_input, ("主题", "topic")) or topic

    return {
        "resume_text": infer_resume_text(user_input),
        "jd_text": infer_jd_text(user_input),
        "topic": topic,
        "target_role": infer_target_role(user_input, memory),
        "has_resume_artifact": bool(artifacts.get("resume_analysis")),
        "has_jd_artifact": bool(artifacts.get("jd_analysis")),
        "has_match_artifact": bool(artifacts.get("match_result")),
    }


def is_complex_agent_request(user_input: str, state: AgentState) -> bool:
    lowered = (user_input or "").lower()
    has_resume = bool(infer_resume_text(user_input)) or any(keyword in lowered for keyword in ("简历", "resume", "cv"))
    has_jd = bool(infer_jd_text(user_input)) or any(keyword in lowered for keyword in ("jd", "岗位", "职位", "job description"))
    has_agent_intent = any(keyword in user_input for keyword in AGENT_COMPLEXITY_KEYWORDS)
    has_interview_plan = "面试" in user_input and any(keyword in user_input for keyword in ("准备", "追问", "复盘", "计划"))
    has_learning_plan = any(keyword in user_input for keyword in ("学习", "路线", "计划", "roadmap", "补项目"))

    return (
        (has_resume and has_jd)
        or (has_resume and has_agent_intent)
        or (has_jd and has_agent_intent)
        or has_interview_plan
        or has_learning_plan
        or ("求职准备" in user_input)
    )


def classify_task_type(user_input: str, state: AgentState) -> str:
    lowered = (user_input or "").lower()

    if any(keyword in user_input for keyword in ("面试", "追问", "模拟面试")):
        return "interview_preparation"
    if any(keyword in user_input for keyword in ("学习", "路线", "roadmap", "计划", "补项目")):
        return "learning_plan"
    if any(keyword in lowered for keyword in ("resume", "cv", "jd", "match")) or any(keyword in user_input for keyword in ("简历", "岗位", "职位", "求职准备")):
        return "resume_jd_preparation"
    return "general_planning"


def tool_call(name: str, arguments: dict) -> dict:
    return {
        "id": f"local_{name}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def build_local_tool_call(user_input: str) -> dict | None:
    lowered = (user_input or "").lower()
    resume_text = extract_labeled_block(user_input, ("简历", "resume", "cv"))
    jd_text = extract_labeled_block(user_input, ("jd", "岗位", "职位", "job description"))

    if resume_text and jd_text:
        return tool_call(
            "match_resume_to_jd",
            {
                "resume_text": resume_text,
                "jd_text": jd_text,
            },
        )

    if resume_text:
        return tool_call("analyze_resume", {"resume_text": resume_text})

    if jd_text:
        return tool_call("analyze_jd", {"jd_text": jd_text})

    if any(keyword in lowered for keyword in ("学习", "路线", "计划", "roadmap")):
        return tool_call(
            "get_learning_plan",
            {"topic": guess_learning_topic(user_input)},
        )

    if any(keyword in lowered for keyword in ("简历", "resume", "cv")) and len(user_input) > 120:
        return tool_call("analyze_resume", {"resume_text": user_input})

    if any(keyword in lowered for keyword in ("jd", "岗位", "职位", "job description")) and len(user_input) > 120:
        return tool_call("analyze_jd", {"jd_text": user_input})

    if any(keyword in lowered for keyword in TECHNICAL_QUERY_KEYWORDS):
        return tool_call("rag_search", {"query": user_input})

    return None


def direct_chat_node(state: AgentState) -> dict:
    answer = call_llm(state.get("messages", []))
    updated_messages = list(state.get("messages", []))
    updated_messages.append({"role": "assistant", "content": answer})
    return {
        "messages": updated_messages,
        "answer": answer,
        "agent_mode": False,
        "used_tool": False,
        "tool_name": None,
        "trace_id": None,
        "sources": [],
        "citations": [],
        "agent_status": "idle",
        "next_action": "",
    }


def plan_with_model_tools_node(state: AgentState) -> dict:
    assistant_message = complete_message(
        messages=state.get("messages", []),
        tools=tool_schemas(),
        tool_choice="auto",
    )
    tool_calls = normalize_tool_calls(assistant_message.get("tool_calls"))
    content = assistant_message.get("content") or ""
    updated_messages = list(state.get("messages", []))

    if tool_calls:
        updated_messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })
        return {
            "messages": updated_messages,
            "tool_calls": tool_calls,
            "route": "execute_tools",
        }

    updated_messages.append({"role": "assistant", "content": content})
    return {
        "messages": updated_messages,
        "answer": content,
        "used_tool": False,
        "tool_name": None,
        "trace_id": None,
        "sources": [],
        "citations": [],
        "route": "finish",
    }


def build_plan_step(step_type: str, title: str, tool_name: str, input_summary: str) -> dict:
    return {
        "step_id": f"{step_type}_{title}_{datetime.now().timestamp()}",
        "step_type": step_type,
        "title": title,
        "tool_name": tool_name,
        "status": "pending",
        "input_summary": input_summary,
        "retries": 0,
    }


def build_agent_plan(user_input: str, task_type: str, context_inputs: dict, artifacts: dict) -> tuple[str, list[dict]]:
    steps = []

    if task_type == "resume_jd_preparation":
        goal = "围绕简历与 JD 生成一套可展示、可复述的求职准备方案。"
        if context_inputs.get("resume_text") and not artifacts.get("resume_analysis"):
            steps.append(build_plan_step("analyze_resume", "分析简历关键词", "analyze_resume", "从用户提供的简历文本提取技能与可强化项。"))
        if context_inputs.get("jd_text") and not artifacts.get("jd_analysis"):
            steps.append(build_plan_step("analyze_jd", "分析 JD 关键词", "analyze_jd", "从岗位描述提取关键词和准备重点。"))
        if context_inputs.get("resume_text") and context_inputs.get("jd_text") and not artifacts.get("match_result"):
            steps.append(build_plan_step("match_resume_to_jd", "计算简历匹配度", "match_resume_to_jd", "对齐简历与 JD，找出已匹配项和缺口。"))
        steps.append(build_plan_step("synthesize_resume_bullets", "生成简历项目 bullet", "synthesize_resume_bullets", "把已有分析结果转成更适合面试展示的项目表述。"))
        steps.append(build_plan_step("prepare_interview_focus", "整理面试重点", "prepare_interview_focus", "围绕岗位关键词和缺口整理追问与答题角度。"))
    elif task_type == "interview_preparation":
        goal = "生成一套能直接用于模拟面试的准备重点与追问清单。"
        if context_inputs.get("jd_text") and not artifacts.get("jd_analysis"):
            steps.append(build_plan_step("analyze_jd", "分析岗位要求", "analyze_jd", "先提取岗位关键词，建立面试主线。"))
        steps.append(build_plan_step("rag_search", "检索相关知识点", "rag_search", "补充 RAG / Agent / 后端相关证据和复习资料。"))
        steps.append(build_plan_step("prepare_interview_focus", "生成面试策略", "prepare_interview_focus", "输出重点、追问和建议答法。"))
    elif task_type == "learning_plan":
        goal = "给出一条两周内能落地展示的学习与补强路径。"
        if context_inputs.get("jd_text") and not artifacts.get("jd_analysis"):
            steps.append(build_plan_step("analyze_jd", "识别岗位缺口", "analyze_jd", "先从岗位要求提取需要补的技术点。"))
        steps.append(build_plan_step("rag_search", "检索核心知识点", "rag_search", "先补当前主题的知识库证据。"))
        steps.append(build_plan_step("generate_learning_plan", "生成学习计划", "generate_learning_plan", "围绕当前主题安排 Demo、复盘和简历产物。"))
        steps.append(build_plan_step("job_search_strategy", "生成冲刺策略", "job_search_strategy", "补齐简历、面试和投递节奏。"))
    else:
        goal = "把当前求职问题拆成更可执行的动作。"
        steps.append(build_plan_step("job_search_strategy", "生成求职策略", "job_search_strategy", "先基于已有信息产出一版可执行计划。"))

    return goal, steps[:MAX_AGENT_STEPS]


def summarize_memory_from_step(memory: dict, step_type: str, result: dict, context_inputs: dict) -> dict:
    updated = ensure_memory(memory)
    candidate = updated["candidate_profile"]
    target = updated["job_target"]
    artifact_memory = updated["artifact_memory"]

    if step_type == "analyze_resume":
        candidate["skills"] = result.get("skills", [])
        candidate["strengths"] = result.get("skills", [])[:5]
        candidate["project_highlights"] = [
            f"可重点讲 {skill} 的项目使用场景。"
            for skill in result.get("skills", [])[:3]
        ]

    if step_type == "analyze_jd":
        target["keywords"] = result.get("keywords", [])
        target["summary"] = "、".join(result.get("keywords", [])[:5])
        target["target_role"] = context_inputs.get("target_role", "")

    if step_type == "match_resume_to_jd":
        candidate["weaknesses"] = result.get("missing_keywords", [])[:5]
        artifact_memory["last_match_score"] = result.get("match_score")

    if step_type == "generate_learning_plan":
        artifact_memory["last_learning_topic"] = result.get("topic", context_inputs.get("topic", ""))

    if step_type == "synthesize_resume_bullets":
        artifact_memory["last_resume_bullets"] = result.get("bullets", [])[:3]

    if step_type == "prepare_interview_focus":
        artifact_memory["last_interview_focus"] = result.get("focus_topics", [])[:3]

    updated["last_updated_at"] = now_ts()
    return updated


def derive_rag_query(context_inputs: dict, artifacts: dict, user_input: str) -> str:
    if context_inputs.get("topic"):
        return f"{context_inputs['topic']} 面试 项目 经验 原理"
    jd_analysis = artifacts.get("jd_analysis", {}) or {}
    keywords = jd_analysis.get("keywords", [])
    if keywords:
        return " ".join(keywords[:3]) + " 面试 项目 原理"
    return user_input[:120]


def build_fallback_match_result(artifacts: dict) -> dict:
    jd_analysis = artifacts.get("jd_analysis", {}) or {}
    return {
        "matched_keywords": [],
        "missing_keywords": jd_analysis.get("keywords", [])[:3],
    }


def build_fallback_jd_analysis(user_input: str) -> dict:
    keywords = extract_tech_keywords(user_input)
    return {
        "keywords": keywords,
        "keyword_count": len(keywords),
        "preparation_focus": [
            f"准备 {keyword} 的项目场景、原理边界和优化方案。"
            for keyword in keywords[:4]
        ] or ["先围绕当前问题里的技术关键词准备项目场景和优化思路。"],
    }


def execute_agent_step(step: dict, state: AgentState) -> dict:
    step_type = step.get("step_type", "")
    context_inputs = state.get("context_inputs", {})
    artifacts = dict(state.get("artifacts", {}) or {})
    memory = ensure_memory(state.get("memory"))

    try:
        if step_type == "analyze_resume":
            resume_text = context_inputs.get("resume_text")
            if not resume_text:
                raise ValueError("当前请求里没有可分析的简历文本。")
            result = analyze_resume_text(resume_text)
            artifact_key = "resume_analysis"
            citations = []
        elif step_type == "analyze_jd":
            jd_text = context_inputs.get("jd_text")
            if not jd_text:
                raise ValueError("当前请求里没有可分析的 JD 文本。")
            result = analyze_jd_text(jd_text)
            artifact_key = "jd_analysis"
            citations = []
        elif step_type == "match_resume_to_jd":
            resume_text = context_inputs.get("resume_text")
            jd_text = context_inputs.get("jd_text")
            if not resume_text or not jd_text:
                raise ValueError("缺少简历或 JD 原文，暂时无法计算匹配度。")
            result = match_resume_to_jd(resume_text, jd_text)
            artifact_key = "match_result"
            citations = []
        elif step_type == "rag_search":
            query = derive_rag_query(context_inputs, artifacts, state.get("user_input", ""))
            sources, retriever = retrieve_sources(query=query, top_k=3)
            citations = format_citations(sources)
            result = {
                "query": query,
                "answer": "已完成知识库检索，可用于后续面试准备和策略生成。",
                "sources": sources,
                "citations": citations,
                "retriever": retriever,
                "summary": f"已围绕 {query} 检索知识库并返回引用来源。",
                "evidence": [
                    f"命中来源数：{len(sources)}",
                    f"检索器：{retriever.get('type', '-')}",
                ],
                "next_actions": ["把命中的知识点转成面试回答和项目复盘。"],
            }
            artifact_key = "rag_result"
            citations = citations
        elif step_type == "generate_learning_plan":
            current_gap = (artifacts.get("match_result", {}) or {}).get("missing_keywords", [])
            result = build_structured_learning_plan(
                topic=context_inputs.get("topic", "AI Agent"),
                current_gap=current_gap,
            )
            artifact_key = "learning_plan"
            citations = []
        elif step_type == "synthesize_resume_bullets":
            resume_analysis = artifacts.get("resume_analysis")
            jd_analysis = artifacts.get("jd_analysis")
            match_result = artifacts.get("match_result")
            if not resume_analysis or not jd_analysis or not match_result:
                raise ValueError("简历 bullet 生成需要先完成简历、JD 和匹配分析。")
            result = synthesize_resume_bullets(resume_analysis, jd_analysis, match_result)
            artifact_key = "resume_bullets"
            citations = []
        elif step_type == "prepare_interview_focus":
            jd_analysis = artifacts.get("jd_analysis", {}) or build_fallback_jd_analysis(state.get("user_input", ""))
            match_result = artifacts.get("match_result") or build_fallback_match_result(artifacts)
            if not jd_analysis:
                raise ValueError("面试重点生成需要先有 JD 分析结果。")
            result = prepare_interview_focus(
                jd_analysis=jd_analysis,
                match_result=match_result,
                rag_result=artifacts.get("rag_result"),
            )
            artifact_key = "interview_focus"
            citations = (artifacts.get("rag_result", {}) or {}).get("citations", [])
        elif step_type == "job_search_strategy":
            current_gap = (artifacts.get("match_result", {}) or {}).get("missing_keywords", [])
            result = job_search_strategy(
                candidate_profile=memory.get("candidate_profile", {}),
                job_target=memory.get("job_target", {}),
                current_gap=current_gap,
            )
            artifact_key = "job_search_strategy"
            citations = []
        else:
            raise ValueError(f"Unknown step type: {step_type}")

        updated_artifacts = dict(artifacts)
        updated_artifacts[artifact_key] = result
        updated_memory = summarize_memory_from_step(memory, step_type, result, context_inputs)

        return {
            "status": "completed",
            "tool_name": step.get("tool_name"),
            "artifact_key": artifact_key,
            "artifact_value": result,
            "artifacts": updated_artifacts,
            "memory": updated_memory,
            "result_summary": build_result_summary(result),
            "citations": citations,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "tool_name": step.get("tool_name"),
            "error": str(exc),
            "artifacts": artifacts,
            "memory": memory,
            "result_summary": str(exc),
            "citations": [],
        }


def review_step_result(step: dict, execution: dict, state: AgentState) -> tuple[str, str]:
    if execution.get("status") == "failed":
        if step.get("retries", 0) < MAX_STEP_RETRIES:
            return "retry", "重试当前步骤，确认是否为临时失败。"
        return "replan", "当前步骤多次失败，跳过并按已有证据继续生成结果。"

    if state.get("current_step_index", 0) >= len(state.get("plan_steps", [])) - 1:
        return "finish", "计划中的关键步骤已完成，整理最终建议。"

    return "continue", "进入下一步，继续补全求职方案。"


def apply_review_to_plan(step: dict, decision: str) -> dict:
    updated = dict(step)
    if decision == "retry":
        updated["retries"] = updated.get("retries", 0) + 1
        updated["status"] = "pending"
    elif decision == "replan":
        updated["status"] = "failed"
    else:
        updated["status"] = "completed"
    return updated


def filter_remaining_steps_after_failure(plan_steps: list[dict], failed_step_type: str) -> list[dict]:
    filtered = []
    skip_map = {
        "analyze_resume": {"match_resume_to_jd", "synthesize_resume_bullets"},
        "analyze_jd": {"match_resume_to_jd", "prepare_interview_focus", "synthesize_resume_bullets"},
        "match_resume_to_jd": {"synthesize_resume_bullets"},
    }
    blocked = skip_map.get(failed_step_type, set())

    for step in plan_steps:
        if step.get("step_type") in blocked and step.get("status") == "pending":
            skipped = dict(step)
            skipped["status"] = "skipped"
            filtered.append(skipped)
            continue
        filtered.append(step)

    return filtered


def render_agent_answer(state: AgentState) -> str:
    artifacts = state.get("artifacts", {}) or {}
    step_history = state.get("step_history", []) or []
    lines = [state.get("user_goal") or "已完成本轮求职 Agent 分析。"]

    match_result = artifacts.get("match_result", {})
    if match_result:
        lines.append(f"当前简历与 JD 的匹配度是 {match_result.get('match_score', 0)}%。")
        matched = match_result.get("matched_keywords", [])
        missing = match_result.get("missing_keywords", [])
        if matched:
            lines.append(f"已匹配关键词：{', '.join(matched[:5])}。")
        if missing:
            lines.append(f"优先补强关键词：{', '.join(missing[:5])}。")

    resume_bullets = artifacts.get("resume_bullets", {})
    if resume_bullets.get("bullets"):
        lines.append("建议优先改简历项目描述：")
        lines.extend(f"- {item}" for item in resume_bullets.get("bullets", [])[:3])

    interview_focus = artifacts.get("interview_focus", {})
    if interview_focus.get("focus_topics"):
        lines.append("面试准备重点：")
        lines.extend(f"- {item}" for item in interview_focus.get("focus_topics", [])[:3])

    learning_plan = artifacts.get("learning_plan", {})
    if learning_plan.get("plan_days"):
        lines.append("学习路径建议：")
        lines.extend(
            f"- Day {item['day']}: {item['focus']} -> {item['deliverable']}"
            for item in learning_plan.get("plan_days", [])[:3]
        )

    strategy = artifacts.get("job_search_strategy", {})
    if strategy.get("application_strategy"):
        lines.append("下一步动作：")
        lines.extend(f"- {item}" for item in strategy.get("application_strategy", [])[:3])
    else:
        lines.append("下一步动作：")
        lines.append(f"- {state.get('next_action') or '继续把结果改成简历 bullet 和面试答案。'}")

    if step_history:
        lines.append("Agent Decision：本轮先做分析，再做匹配与复盘，最后整理成可执行建议。")

    return "\n\n".join(lines)


def current_step_payload(state: AgentState) -> dict | None:
    plan_steps = state.get("plan_steps", []) or []
    index = state.get("current_step_index", 0)
    if index < 0 or index >= len(plan_steps):
        return None
    return plan_steps[index]


def route_turn(state: AgentState) -> dict:
    user_input = state.get("user_input", "")
    provider = get_provider()
    context_inputs = collect_context_inputs(user_input, state)

    if is_complex_agent_request(user_input, state):
        return {
            "route": "build_goal_and_plan",
            "agent_mode": True,
            "task_type": classify_task_type(user_input, state),
            "context_inputs": context_inputs,
            "agent_status": "planning",
        }

    if provider == "codex_cli":
        local_tool_call = build_local_tool_call(user_input)
        if local_tool_call:
            return {
                "route": "execute_tools",
                "tool_calls": [local_tool_call],
                "context_inputs": context_inputs,
            }
        return {"route": "direct_chat", "context_inputs": context_inputs}

    local_tool_call = build_local_tool_call(user_input)
    if local_tool_call:
        return {
            "route": "execute_tools",
            "tool_calls": [local_tool_call],
            "context_inputs": context_inputs,
        }

    if should_offer_tools(user_input):
        return {"route": "plan_with_model_tools", "context_inputs": context_inputs}

    return {"route": "direct_chat", "context_inputs": context_inputs}


def build_goal_and_plan_node(state: AgentState) -> dict:
    task_type = state.get("task_type") or classify_task_type(state.get("user_input", ""), state)
    context_inputs = state.get("context_inputs") or collect_context_inputs(state.get("user_input", ""), state)
    goal, plan_steps = build_agent_plan(
        user_input=state.get("user_input", ""),
        task_type=task_type,
        context_inputs=context_inputs,
        artifacts=state.get("artifacts", {}) or {},
    )

    updated_traces = list(state.get("traces", []))
    planner_payload = {
        "task_type": task_type,
        "goal": goal,
        "plan_steps": [
            {
                "step_type": step["step_type"],
                "title": step["title"],
                "tool_name": step["tool_name"],
            }
            for step in plan_steps
        ],
    }
    append_trace(
        traces=updated_traces,
        tool_name="build_goal_and_plan",
        arguments={"user_input": state.get("user_input", "")[:200]},
        tool_result=planner_payload,
        metadata={"actor": "planner"},
        mode="planner",
    )

    return {
        "agent_mode": True,
        "task_type": task_type,
        "user_goal": goal,
        "plan_steps": plan_steps,
        "current_step_index": 0,
        "step_history": [],
        "agent_status": "executing",
        "next_action": "从第一步开始执行。",
        "traces": updated_traces,
    }


def execute_current_step_node(state: AgentState) -> dict:
    plan_steps = [dict(step) for step in (state.get("plan_steps", []) or [])]
    current_index = state.get("current_step_index", 0)
    updated_traces = list(state.get("traces", []))
    step_history = list(state.get("step_history", []))
    step = plan_steps[current_index]

    running_step = dict(step)
    running_step["status"] = "running"
    plan_steps[current_index] = running_step

    execution = execute_agent_step(running_step, state)
    step_record = {
        "step_index": current_index + 1,
        "step_type": running_step.get("step_type"),
        "title": running_step.get("title"),
        "tool_name": execution.get("tool_name"),
        "status": execution.get("status"),
        "input_summary": running_step.get("input_summary", ""),
        "result_summary": execution.get("result_summary", ""),
        "error": execution.get("error"),
        "citations": execution.get("citations", []),
        "created_at": now_ts(),
    }
    step_history.append(step_record)

    trace_id = append_trace(
        traces=updated_traces,
        tool_name=running_step.get("tool_name", running_step.get("step_type", "agent_step")),
        arguments={
            "step_type": running_step.get("step_type"),
            "title": running_step.get("title"),
        },
        tool_result=execution.get("artifact_value", execution.get("error", execution.get("result_summary"))),
        metadata={
            "actor": "executor",
            "status": execution.get("status"),
            "citations": execution.get("citations", []),
        },
        mode="executor",
    )

    return {
        "plan_steps": plan_steps,
        "step_history": step_history,
        "artifacts": execution.get("artifacts", state.get("artifacts", {})),
        "memory": execution.get("memory", state.get("memory", {})),
        "tool_name": running_step.get("tool_name"),
        "trace_id": trace_id,
        "sources": (execution.get("artifact_value", {}) or {}).get("sources", []),
        "citations": execution.get("citations", []),
        "agent_status": "reviewing",
        "next_action": execution.get("result_summary", ""),
        "traces": updated_traces,
    }


def review_step_result_node(state: AgentState) -> dict:
    step_history = state.get("step_history", []) or []
    if not step_history:
        return {"review_decision": "finish", "next_action": "没有可复盘的步骤，直接结束。"}

    last_step = step_history[-1]
    step = (state.get("plan_steps", []) or [])[state.get("current_step_index", 0)]
    decision, reason = review_step_result(step, last_step, state)

    updated_traces = list(state.get("traces", []))
    append_trace(
        traces=updated_traces,
        tool_name="review_step_result",
        arguments={"step_type": last_step.get("step_type"), "status": last_step.get("status")},
        tool_result={"decision": decision, "reason": reason},
        metadata={"actor": "reviewer"},
        mode="reviewer",
    )

    step_history[-1]["review_decision"] = decision
    step_history[-1]["next_action"] = reason

    return {
        "step_history": step_history,
        "review_decision": decision,
        "next_action": reason,
        "agent_status": "reviewed",
        "traces": updated_traces,
    }


def replan_if_needed_node(state: AgentState) -> dict:
    plan_steps = [dict(step) for step in (state.get("plan_steps", []) or [])]
    current_index = state.get("current_step_index", 0)
    decision = state.get("review_decision", "finish")

    current_step = plan_steps[current_index]
    plan_steps[current_index] = apply_review_to_plan(current_step, decision)

    if decision == "retry":
        return {
            "plan_steps": plan_steps,
            "current_step_index": current_index,
            "agent_status": "executing",
        }

    if decision == "replan":
        plan_steps = filter_remaining_steps_after_failure(plan_steps, current_step.get("step_type", ""))
        next_index = current_index + 1
        while next_index < len(plan_steps) and plan_steps[next_index].get("status") == "skipped":
            next_index += 1
        return {
            "plan_steps": plan_steps,
            "current_step_index": next_index,
            "agent_status": "executing" if next_index < len(plan_steps) else "completed",
        }

    if decision == "continue":
        next_index = current_index + 1
        return {
            "plan_steps": plan_steps,
            "current_step_index": next_index,
            "agent_status": "executing" if next_index < len(plan_steps) else "completed",
        }

    return {
        "plan_steps": plan_steps,
        "agent_status": "completed",
    }


def finalize_answer_node(state: AgentState) -> dict:
    plan_steps = [dict(step) for step in (state.get("plan_steps", []) or [])]
    current_index = state.get("current_step_index", 0)
    if 0 <= current_index < len(plan_steps) and state.get("review_decision") == "finish":
        final_step = dict(plan_steps[current_index])
        final_step["status"] = "completed"
        plan_steps[current_index] = final_step

    answer = render_agent_answer(state)
    updated_messages = list(state.get("messages", []))
    updated_messages.append({"role": "assistant", "content": answer})

    current = plan_steps[current_index] if 0 <= current_index < len(plan_steps) else None
    final_summary = state.get("next_action") or "已完成本轮 Agent 执行。"

    return {
        "messages": updated_messages,
        "answer": answer,
        "used_tool": bool(state.get("step_history")),
        "agent_mode": True,
        "agent_status": "completed",
        "current_step_index": state.get("current_step_index", 0),
        "plan_steps": plan_steps,
        "current_step": current,
        "final_summary": final_summary,
        "context_inputs": {
            "topic": (state.get("context_inputs", {}) or {}).get("topic", ""),
            "resume_present": bool((state.get("context_inputs", {}) or {}).get("resume_text")),
            "jd_present": bool((state.get("context_inputs", {}) or {}).get("jd_text")),
            "target_role": (state.get("context_inputs", {}) or {}).get("target_role", ""),
        },
    }


def summarize_tools_node(state: AgentState) -> dict:
    answer = ""

    if get_provider() == "openai_api":
        assistant_message = complete_message(messages=state.get("messages", []))
        answer = assistant_message.get("content") or ""

    if not answer:
        answer = summarize_tool_results(
            user_input=state.get("user_input", ""),
            tool_results=state.get("tool_results", []),
        )

    updated_messages = list(state.get("messages", []))
    updated_messages.append({"role": "assistant", "content": answer})
    return {
        "messages": updated_messages,
        "answer": answer,
    }


def execute_tools_node(state: AgentState) -> dict:
    updated_messages = list(state.get("messages", []))
    updated_traces = list(state.get("traces", []))
    tool_results = []
    first_trace_id = None
    first_tool_name = None
    sources = []
    citations = []

    for tool_call_item in state.get("tool_calls", []):
        tool_name = tool_call_item.get("function", {}).get("name")
        arguments = parse_tool_arguments(tool_call_item.get("function", {}).get("arguments"))
        execution = execute_tool_with_metadata(tool_name, arguments)
        trace_id = append_trace(
            traces=updated_traces,
            tool_name=tool_name,
            arguments=arguments,
            tool_result=execution["result"],
            metadata=execution["metadata"],
            mode="langgraph",
        )

        if first_trace_id is None:
            first_trace_id = trace_id
            first_tool_name = tool_name
            sources = execution["metadata"].get("sources", [])
            citations = execution["metadata"].get("citations", [])

        payload = {
            "tool_name": tool_name,
            "result": execution["result"],
            "metadata": execution["metadata"],
        }
        tool_results.append(payload)
        updated_messages.append({
            "role": "tool",
            "tool_call_id": tool_call_item.get("id", f"call_{trace_id}"),
            "content": json.dumps(payload, ensure_ascii=False),
        })

    return {
        "messages": updated_messages,
        "tool_results": tool_results,
        "traces": updated_traces,
        "used_tool": bool(tool_results),
        "tool_name": first_tool_name,
        "trace_id": first_trace_id,
        "sources": sources,
        "citations": citations,
        "agent_mode": False,
        "agent_status": "idle",
    }


def route_after_model_tools(state: AgentState) -> str:
    return "execute_tools" if state.get("tool_calls") else "finish"


def route_after_entry(state: AgentState) -> str:
    return state.get("route", "direct_chat")


def route_after_review(state: AgentState) -> str:
    if state.get("review_decision") == "finish":
        return "finalize_answer"
    return "replan_if_needed"


def route_after_replan(state: AgentState) -> str:
    if state.get("current_step_index", 0) < len(state.get("plan_steps", []) or []):
        return "execute_current_step"
    return "finalize_answer"


@lru_cache(maxsize=1)
def get_agent_graph():
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is not installed. Please install langgraph and langchain-core from requirements.txt."
        ) from exc

    graph = StateGraph(AgentState)
    graph.add_node("route_turn", route_turn)
    graph.add_node("direct_chat", direct_chat_node)
    graph.add_node("plan_with_model_tools", plan_with_model_tools_node)
    graph.add_node("build_goal_and_plan", build_goal_and_plan_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("execute_current_step", execute_current_step_node)
    graph.add_node("review_step_result", review_step_result_node)
    graph.add_node("replan_if_needed", replan_if_needed_node)
    graph.add_node("finalize_answer", finalize_answer_node)
    graph.add_node("summarize_tools", summarize_tools_node)

    graph.add_edge(START, "route_turn")
    graph.add_conditional_edges(
        "route_turn",
        route_after_entry,
        {
            "direct_chat": "direct_chat",
            "plan_with_model_tools": "plan_with_model_tools",
            "execute_tools": "execute_tools",
            "build_goal_and_plan": "build_goal_and_plan",
        },
    )
    graph.add_conditional_edges(
        "plan_with_model_tools",
        route_after_model_tools,
        {
            "execute_tools": "execute_tools",
            "finish": END,
        },
    )
    graph.add_edge("build_goal_and_plan", "execute_current_step")
    graph.add_edge("execute_current_step", "review_step_result")
    graph.add_conditional_edges(
        "review_step_result",
        route_after_review,
        {
            "replan_if_needed": "replan_if_needed",
            "finalize_answer": "finalize_answer",
        },
    )
    graph.add_conditional_edges(
        "replan_if_needed",
        route_after_replan,
        {
            "execute_current_step": "execute_current_step",
            "finalize_answer": "finalize_answer",
        },
    )
    graph.add_edge("direct_chat", END)
    graph.add_edge("execute_tools", "summarize_tools")
    graph.add_edge("summarize_tools", END)
    graph.add_edge("finalize_answer", END)
    return graph.compile(checkpointer=get_checkpointer())


def get_thread_state(session_id: str) -> dict:
    graph = get_agent_graph()
    snapshot = graph.get_state(get_thread_config(session_id))
    if not snapshot or not getattr(snapshot, "values", None):
        return {}
    return snapshot.values


def update_thread_state(session_id: str, values: dict) -> None:
    if not values:
        return

    graph = get_agent_graph()
    graph.update_state(get_thread_config(session_id), values)


def clear_thread_state(session_id: str) -> None:
    checkpointer = get_checkpointer()
    try:
        checkpointer.delete_thread(session_id)
    except Exception:
        pass


def build_agent_response_payload(result: dict) -> dict:
    current_step = current_step_payload(result)
    return {
        "agent_mode": bool(result.get("agent_mode")),
        "goal": result.get("user_goal", ""),
        "task_type": result.get("task_type", ""),
        "plan": result.get("plan_steps", []),
        "current_step": current_step,
        "step_history": result.get("step_history", []),
        "agent_status": result.get("agent_status", "idle"),
        "next_action": result.get("next_action", ""),
        "artifacts": result.get("artifacts", {}),
        "final_summary": result.get("final_summary", ""),
    }


def run_langgraph_agent_turn(
    user_input: str,
    messages: list,
    traces: list,
    session_data: dict | None = None,
    session_id: str = "default",
) -> dict:
    graph = get_agent_graph()
    initial_messages = [dict(message) for message in messages]
    initial_messages.append({"role": "user", "content": user_input})
    session_data = session_data or {}

    state: AgentState = {
        "messages": initial_messages,
        "user_input": user_input,
        "traces": list(traces),
        "tool_calls": [],
        "tool_results": [],
        "used_tool": False,
        "tool_name": None,
        "trace_id": None,
        "sources": [],
        "citations": [],
        "answer": "",
        "agent_mode": False,
        "user_goal": "",
        "task_type": "",
        "context_inputs": {},
        "plan_steps": [],
        "current_step_index": 0,
        "step_history": [],
        "artifacts": dict((session_data.get("agent") or {}).get("artifacts", {})),
        "memory": ensure_memory(session_data.get("memory")),
        "agent_status": "idle",
        "review_decision": "",
        "next_action": "",
        "final_summary": "",
    }
    result = graph.invoke(state, config=get_thread_config(session_id))

    messages.clear()
    messages.extend(result.get("messages", []))

    traces.clear()
    traces.extend(result.get("traces", []))

    return {
        "answer": result.get("answer", ""),
        "used_tool": result.get("used_tool", False),
        "tool_name": result.get("tool_name"),
        "trace_id": result.get("trace_id"),
        "sources": result.get("sources", []),
        "citations": result.get("citations", []),
        "memory": ensure_memory(result.get("memory")),
        "agent": build_agent_response_payload(result),
    }


def show_traces(traces: list) -> None:
    if not traces:
        print("No traces yet.")
        return

    for index, trace in enumerate(traces, start=1):
        print(f"Trace {index}")
        print(f"Tool: {trace.get('tool_name')}")
        print(f"Arguments: {trace.get('arguments')}")
        print("Result:")
        print(trace.get("result"))
        print("-" * 40)
