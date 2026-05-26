from typing import Any

from typing_extensions import Literal, TypedDict


RouteName = Literal[
    "direct_chat",
    "plan_with_model_tools",
    "execute_tools",
    "finish",
]


class AgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    user_input: str
    interview: dict[str, Any]
    route: RouteName
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    traces: list[dict[str, Any]]
    used_tool: bool
    tool_name: str | None
    trace_id: int | None
    sources: list[Any]
    citations: list[Any]
    answer: str
