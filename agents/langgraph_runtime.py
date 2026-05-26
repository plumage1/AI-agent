import json
import re
from datetime import datetime
from functools import lru_cache

from agents.langgraph_state import AgentState
from core.langgraph_checkpoint import get_checkpointer, get_thread_config
from core.llm import call_llm, complete_message, get_provider
from prompts.tool_summary_prompt import TOOL_SUMMARY_PROMPT
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
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    return len(traces) - 1


def execute_tool_with_metadata(tool_name: str, arguments: dict) -> dict:
    metadata = {}

    if tool_name == "rag_search":
        from rag.rag_chain import answer_with_rag_and_sources

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
        return {
            "result": tool_func(**(arguments or {})),
            "metadata": metadata,
        }
    except TypeError as exc:
        return {
            "result": f"Tool {tool_name} received invalid arguments: {exc}",
            "metadata": metadata,
        }


def route_turn(state: AgentState) -> dict:
    user_input = state.get("user_input", "")
    provider = get_provider()

    if provider == "codex_cli":
        local_tool_call = build_local_tool_call(user_input)
        if local_tool_call:
            return {
                "route": "execute_tools",
                "tool_calls": [local_tool_call],
            }
        return {"route": "direct_chat"}

    if should_offer_tools(user_input):
        return {"route": "plan_with_model_tools"}

    return {"route": "direct_chat"}


def direct_chat_node(state: AgentState) -> dict:
    answer = call_llm(state.get("messages", []))
    updated_messages = list(state.get("messages", []))
    updated_messages.append({"role": "assistant", "content": answer})
    return {
        "messages": updated_messages,
        "answer": answer,
        "used_tool": False,
        "tool_name": None,
        "trace_id": None,
        "sources": [],
        "citations": [],
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


def execute_tools_node(state: AgentState) -> dict:
    updated_messages = list(state.get("messages", []))
    updated_traces = list(state.get("traces", []))
    tool_results = []
    first_trace_id = None
    first_tool_name = None
    sources = []
    citations = []

    for tool_call in state.get("tool_calls", []):
        tool_name = tool_call.get("function", {}).get("name")
        arguments = parse_tool_arguments(tool_call.get("function", {}).get("arguments"))
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
            "tool_call_id": tool_call.get("id", f"call_{trace_id}"),
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


def route_after_model_tools(state: AgentState) -> str:
    return "execute_tools" if state.get("tool_calls") else "finish"


def route_after_entry(state: AgentState) -> str:
    return state.get("route", "direct_chat")


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


def tool_call(name: str, arguments: dict) -> dict:
    return {
        "id": f"local_{name}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
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
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("summarize_tools", summarize_tools_node)

    graph.add_edge(START, "route_turn")
    graph.add_conditional_edges(
        "route_turn",
        route_after_entry,
        {
            "direct_chat": "direct_chat",
            "plan_with_model_tools": "plan_with_model_tools",
            "execute_tools": "execute_tools",
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
    graph.add_edge("direct_chat", END)
    graph.add_edge("execute_tools", "summarize_tools")
    graph.add_edge("summarize_tools", END)
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


def run_langgraph_agent_turn(
    user_input: str,
    messages: list,
    traces: list,
    session_id: str = "default",
) -> dict:
    graph = get_agent_graph()
    initial_messages = [dict(message) for message in messages]
    initial_messages.append({"role": "user", "content": user_input})

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
