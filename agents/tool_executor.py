from datetime import datetime

from core.llm import call_llm
from prompts.tool_summary_prompt import TOOL_SUMMARY_PROMPT
from tools.registry import TOOLS


def summarize_tool_result(user_input: str, tool_result: str) -> str:
    messages = [
        {"role": "system", "content": TOOL_SUMMARY_PROMPT},
        {
            "role": "user",
            "content": f"""
用户问题：
{user_input}

工具返回结果：
{tool_result}
"""
        }
    ]

    return call_llm(messages)


def run_tool(user_input: str, route: dict, messages: list, traces: list) -> str:
    tool_name = route.get("tool_name")
    arguments = route.get("arguments", {}) or {}
    metadata = {}

    if tool_name == "rag_search":
        from rag.rag_chain import answer_with_rag_and_sources

        rag_result = answer_with_rag_and_sources(arguments.get("query", ""))
        tool_result = rag_result["answer"]
        metadata["sources"] = rag_result["sources"]
        metadata["citations"] = rag_result.get("citations", [])
        metadata["source_count"] = len(rag_result["sources"])
        metadata["retriever"] = rag_result.get("retriever", {})
    else:
        tool_result = execute_tool(tool_name, arguments)

    final_answer = summarize_tool_result(user_input, tool_result)

    traces.append({
        "tool_name": tool_name,
        "arguments": arguments,
        "result": tool_result,
        "metadata": metadata,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })

    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": final_answer})

    return final_answer


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


def execute_tool(tool_name: str, arguments: dict) -> str:
    if tool_name not in TOOLS:
        return f"工具 {tool_name} 不存在，无法执行。"

    if arguments is None:
        arguments = {}

    try:
        tool_func = TOOLS[tool_name]["function"]
        return tool_func(**arguments)
    except TypeError as e:
        return f"工具 {tool_name} 参数错误：{e}"
