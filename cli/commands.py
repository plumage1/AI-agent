from prompts.job_coach_prompt import SYSTEM_PROMPT
from tools.registry import TOOLS
from agents.tool_executor import show_traces

def show_tools() -> None:
    print("Registered tools:")

    for tool_name, tool_info in TOOLS.items():
        print(f"- {tool_name}: {tool_info['description']}")


def handle_command(command: str, messages: list, traces: list, state: dict) -> str:
    if command == "exit":
        return "exit"

    if command == "clear":
        messages.clear()
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        print("Memory Cleared")
        return "handled"

    if command == "trace":
        show_traces(traces)
        return "handled"

    if command == "debug":
        state["debug"] = not state["debug"]
        print(f"Debug mode: {state['debug']}")
        return "handled"

    if command == "help":
        print("""
Available commands:
- exit: quit the program
- clear: clear conversation memory
- trace: show tool execution traces
- tools: show registered tools
- debug: toggle router debug output
- help: show this help message
""")
        return "handled"

    if command == "tools":
        show_tools()
        return "handled"

    return "none"