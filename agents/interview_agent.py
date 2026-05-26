import json
from datetime import datetime

from core.llm import call_llm
from prompts.interview_prompt import (
    INTERVIEW_EVALUATION_PROMPT,
    INTERVIEW_QUESTION_PROMPT,
)
from rag.rag_chain import build_context, format_citations, retrieve_sources


DEFAULT_INTERVIEW_TOPIC = "AI Agent 项目开发"


def get_interview_state(session: dict) -> dict:
    return session.setdefault("interview", {
        "topic": DEFAULT_INTERVIEW_TOPIC,
        "difficulty": "中等",
        "current_question": None,
        "turns": [],
        "started_at": datetime.now().isoformat(timespec="seconds"),
    })


def parse_json_response(content: str) -> dict:
    try:
        content = content.strip()

        if content.startswith("```json"):
            content = content.replace("```json", "", 1)
            content = content.replace("```", "")
            content = content.strip()

        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "score": 0,
            "feedback": "模型没有返回合法 JSON，建议重新回答或重试。",
            "reference_answer": content,
            "follow_up_question": "请你重新用项目实践角度回答刚才的问题。"
        }


def average_score(turns: list[dict]) -> float:
    if not turns:
        return 0.0

    return round(
        sum(turn.get("score", 0) for turn in turns) / len(turns),
        2,
    )


def generate_interview_question(topic: str, difficulty: str) -> dict:
    sources, retriever = retrieve_sources(topic, top_k=2)
    context = build_context(sources)

    messages = [
        {"role": "system", "content": INTERVIEW_QUESTION_PROMPT},
        {
            "role": "user",
            "content": f"""
练习主题：
{topic}

难度：
{difficulty}

可参考的知识库内容：
{context}

请生成第一道面试问题。
"""
        }
    ]

    question = call_llm(messages)

    return {
        "question": question.strip(),
        "sources": sources,
        "citations": format_citations(sources),
        "retriever": retriever,
    }


def start_interview(
    session: dict,
    topic: str = DEFAULT_INTERVIEW_TOPIC,
    difficulty: str = "中等",
) -> dict:
    question_result = generate_interview_question(topic, difficulty)

    session["interview"] = {
        "topic": topic,
        "difficulty": difficulty,
        "current_question": question_result["question"],
        "turns": [],
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "last_retriever": question_result["retriever"],
        "last_citations": question_result["citations"],
    }

    return {
        "topic": topic,
        "difficulty": difficulty,
        "question": question_result["question"],
        "citations": question_result["citations"],
        "retriever": question_result["retriever"],
    }


def submit_interview_answer(session: dict, answer: str) -> dict:
    state = get_interview_state(session)

    if not state.get("current_question"):
        start_result = start_interview(
            session=session,
            topic=state.get("topic", DEFAULT_INTERVIEW_TOPIC),
            difficulty=state.get("difficulty", "中等"),
        )
        state = get_interview_state(session)
        current_question = start_result["question"]
    else:
        current_question = state["current_question"]

    sources, retriever = retrieve_sources(current_question, top_k=2)
    context = build_context(sources)

    messages = [
        {"role": "system", "content": INTERVIEW_EVALUATION_PROMPT},
        {
            "role": "user",
            "content": f"""
面试主题：
{state.get("topic")}

当前问题：
{current_question}

候选人回答：
{answer}

可参考的知识库内容：
{context}

请完成评分、反馈、参考答案和下一轮追问。
"""
        }
    ]

    evaluation = parse_json_response(call_llm(messages))

    turn = {
        "question": current_question,
        "answer": answer,
        "score": int(evaluation.get("score", 0)),
        "feedback": evaluation.get("feedback", ""),
        "reference_answer": evaluation.get("reference_answer", ""),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "citations": format_citations(sources),
    }

    state["turns"].append(turn)
    state["current_question"] = evaluation.get("follow_up_question", "")
    state["last_retriever"] = retriever
    state["last_citations"] = format_citations(sources)

    return {
        "score": turn["score"],
        "feedback": turn["feedback"],
        "reference_answer": turn["reference_answer"],
        "follow_up_question": state["current_question"],
        "turn_count": len(state["turns"]),
        "average_score": average_score(state["turns"]),
        "citations": turn["citations"],
        "retriever": retriever,
    }


def summarize_interview(session: dict) -> dict:
    state = get_interview_state(session)
    turns = state.get("turns", [])

    return {
        "topic": state.get("topic"),
        "difficulty": state.get("difficulty"),
        "current_question": state.get("current_question"),
        "turn_count": len(turns),
        "average_score": average_score(turns),
        "turns": turns,
        "last_citations": state.get("last_citations", []),
        "last_retriever": state.get("last_retriever", {}),
    }


def reset_interview(session: dict) -> None:
    session.pop("interview", None)
