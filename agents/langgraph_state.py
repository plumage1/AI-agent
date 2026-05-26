from typing import Any

from typing_extensions import Literal, TypedDict


RouteName = Literal[
    "direct_chat",
    "plan_with_model_tools",
    "execute_tools",
    "build_goal_and_plan",
    "finish",
]


class AgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    user_input: str
    interview: dict[str, Any]
    route: RouteName
    agent_mode: bool
    user_goal: str
    task_type: str
    context_inputs: dict[str, Any]
    plan_steps: list[dict[str, Any]]
    current_step_index: int
    step_history: list[dict[str, Any]]
    artifacts: dict[str, Any]
    memory: dict[str, Any]
    agent_status: str
    review_decision: str
    next_action: str
    final_summary: str
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    traces: list[dict[str, Any]]
    used_tool: bool
    tool_name: str | None
    trace_id: int | None
    sources: list[Any]
    citations: list[Any]
    answer: str
