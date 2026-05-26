import json
from datetime import datetime
from functools import lru_cache
from typing import Any

from core.llm import call_llm
from prompts.interview_prompt import (
    INTERVIEW_EVALUATION_PROMPT,
    INTERVIEW_QUESTION_PROMPT,
)
from rag.rag_chain import build_context, format_citations, retrieve_sources
from typing_extensions import TypedDict


DEFAULT_INTERVIEW_TOPIC = "AI Agent 项目开发"


class InterviewGraphState(TypedDict, total=False):
    session: dict[str, Any]
    topic: str
    difficulty: str
    answer: str
    current_question: str
    sources: list[dict[str, Any]]
    citations: list[str]
    retriever: dict[str, Any]
    context: str
    question: str
    evaluation: dict[str, Any]
    result: dict[str, Any]


def get_interview_state(session: dict) -> dict:
    if not session.get("interview"):
        session["interview"] = {
            "topic": DEFAULT_INTERVIEW_TOPIC,
            "difficulty": "中等",
            "current_question": None,
            "turns": [],
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
    return session["interview"]


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


def start_interview(
    session: dict,
    topic: str = DEFAULT_INTERVIEW_TOPIC,
    difficulty: str = "中等",
) -> dict:
    graph = get_interview_start_graph()
    state: InterviewGraphState = {
        "session": session,
        "topic": topic,
        "difficulty": difficulty,
    }
    result = graph.invoke(state)
    return result["result"]


def submit_interview_answer(session: dict, answer: str) -> dict:
    graph = get_interview_answer_graph()
    state: InterviewGraphState = {
        "session": session,
        "answer": answer,
    }
    result = graph.invoke(state)
    return result["result"]


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


def build_question_context_node(state: InterviewGraphState) -> dict:
    sources, retriever = retrieve_sources(state["topic"], top_k=2)
    citations = format_citations(sources)
    return {
        "sources": sources,
        "retriever": retriever,
        "citations": citations,
        "context": build_context(sources),
    }


def generate_question_node(state: InterviewGraphState) -> dict:
    messages = [
        {"role": "system", "content": INTERVIEW_QUESTION_PROMPT},
        {
            "role": "user",
            "content": f"""
练习主题：
{state["topic"]}

难度：
{state["difficulty"]}

可参考的知识库内容：
{state["context"]}

请生成第一道面试问题。
"""
        },
    ]
    return {"question": call_llm(messages).strip()}


def finalize_interview_start_node(state: InterviewGraphState) -> dict:
    session = state["session"]
    session["interview"] = {
        "topic": state["topic"],
        "difficulty": state["difficulty"],
        "current_question": state["question"],
        "turns": [],
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "last_retriever": state["retriever"],
        "last_citations": state["citations"],
    }
    return {
        "result": {
            "topic": state["topic"],
            "difficulty": state["difficulty"],
            "question": state["question"],
            "citations": state["citations"],
            "retriever": state["retriever"],
        }
    }


def ensure_interview_question_node(state: InterviewGraphState) -> dict:
    session = state["session"]
    interview_state = get_interview_state(session)

    if not interview_state.get("current_question"):
        question_result = start_interview(
            session=session,
            topic=interview_state.get("topic", DEFAULT_INTERVIEW_TOPIC),
            difficulty=interview_state.get("difficulty", "中等"),
        )
        interview_state = get_interview_state(session)
        return {
            "topic": interview_state.get("topic", DEFAULT_INTERVIEW_TOPIC),
            "difficulty": interview_state.get("difficulty", "中等"),
            "current_question": question_result["question"],
        }

    return {
        "topic": interview_state.get("topic", DEFAULT_INTERVIEW_TOPIC),
        "difficulty": interview_state.get("difficulty", "中等"),
        "current_question": interview_state["current_question"],
    }


def build_answer_context_node(state: InterviewGraphState) -> dict:
    sources, retriever = retrieve_sources(state["current_question"], top_k=2)
    citations = format_citations(sources)
    return {
        "sources": sources,
        "retriever": retriever,
        "citations": citations,
        "context": build_context(sources),
    }


def evaluate_answer_node(state: InterviewGraphState) -> dict:
    messages = [
        {"role": "system", "content": INTERVIEW_EVALUATION_PROMPT},
        {
            "role": "user",
            "content": f"""
面试主题：
{state["topic"]}

当前问题：
{state["current_question"]}

候选人回答：
{state["answer"]}

可参考的知识库内容：
{state["context"]}

请完成评分、反馈、参考答案和下一轮追问。
"""
        },
    ]
    return {"evaluation": parse_json_response(call_llm(messages))}


def finalize_interview_answer_node(state: InterviewGraphState) -> dict:
    session = state["session"]
    interview_state = get_interview_state(session)
    evaluation = state["evaluation"]

    turn = {
        "question": state["current_question"],
        "answer": state["answer"],
        "score": int(evaluation.get("score", 0)),
        "feedback": evaluation.get("feedback", ""),
        "reference_answer": evaluation.get("reference_answer", ""),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "citations": state["citations"],
    }

    interview_state["turns"].append(turn)
    interview_state["current_question"] = evaluation.get("follow_up_question", "")
    interview_state["last_retriever"] = state["retriever"]
    interview_state["last_citations"] = state["citations"]

    return {
        "result": {
            "score": turn["score"],
            "feedback": turn["feedback"],
            "reference_answer": turn["reference_answer"],
            "follow_up_question": interview_state["current_question"],
            "turn_count": len(interview_state["turns"]),
            "average_score": average_score(interview_state["turns"]),
            "citations": turn["citations"],
            "retriever": state["retriever"],
        }
    }


@lru_cache(maxsize=1)
def get_interview_start_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(InterviewGraphState)
    graph.add_node("build_question_context", build_question_context_node)
    graph.add_node("generate_question", generate_question_node)
    graph.add_node("finalize_start", finalize_interview_start_node)
    graph.add_edge(START, "build_question_context")
    graph.add_edge("build_question_context", "generate_question")
    graph.add_edge("generate_question", "finalize_start")
    graph.add_edge("finalize_start", END)
    return graph.compile()


@lru_cache(maxsize=1)
def get_interview_answer_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(InterviewGraphState)
    graph.add_node("ensure_question", ensure_interview_question_node)
    graph.add_node("build_answer_context", build_answer_context_node)
    graph.add_node("evaluate_answer", evaluate_answer_node)
    graph.add_node("finalize_answer", finalize_interview_answer_node)
    graph.add_edge(START, "ensure_question")
    graph.add_edge("ensure_question", "build_answer_context")
    graph.add_edge("build_answer_context", "evaluate_answer")
    graph.add_edge("evaluate_answer", "finalize_answer")
    graph.add_edge("finalize_answer", END)
    return graph.compile()
