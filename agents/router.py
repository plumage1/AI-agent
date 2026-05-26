import json

from core.llm import call_llm
from tools.registry import TOOLS


def build_tools_description() -> str:
    lines = []

    for tool_name, tool_info in TOOLS.items():
        lines.append(f"- {tool_name}")
        lines.append(f"  用途：{tool_info['description']}")
        lines.append("  参数：")

        for param_name, param_desc in tool_info["parameters"].items():
            lines.append(f"    - {param_name}: {param_desc}")

    return "\n".join(lines)


def build_router_prompt() -> str:
    tools_description = build_tools_description()

    return f"""
你是一个 Agent 路由器。
你的任务是判断用户输入是否需要调用工具。

当前可用工具：
{tools_description}

判断规则：
1. 用户要学习路线、学习计划、学习步骤时，调用 get_learning_plan
2. 用户提供简历内容并要求分析、优化、提取技能时，调用 analyze_resume
3. 用户提供 JD 并要求分析岗位、提取关键词时，调用 analyze_jd
4. 用户询问技术概念、面试八股、知识库内容时，调用 rag_search
5. 普通寒暄、闲聊、上下文追问，不调用工具

你只能输出 JSON，不要输出任何解释。

输出格式：
{{
  "use_tool": true 或 false,
  "tool_name": "工具名或 null",
  "arguments": {{}}
}}
"""


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
            "use_tool": False,
            "tool_name": None,
            "arguments": {}
        }


def route_user_input(user_input: str, debug: bool = False) -> dict:
    router_messages = [
        {"role": "system", "content": build_router_prompt()},
        {"role": "user", "content": user_input}
    ]

    content = call_llm(router_messages)

    if debug:
        print("ROUTER RAW:", content)

    route = parse_json_response(content)
    return route
