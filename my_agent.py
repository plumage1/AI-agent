from agents.tool_executor import run_tool, show_traces
from prompts.job_coach_prompt import SYSTEM_PROMPT
from agents.router import route_user_input
from cli.commands import handle_command
from agents.chat_agent import chat_with_memory

def main():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    traces = []
    state = {
        "debug": False
    }

    while True:
        myinput = input("Enter your question: ")
        command = myinput.strip().lower()

        command_result = handle_command(command, messages, traces, state)

        if command_result == "exit":
            break

        if command_result == "handled":
            continue

        route = route_user_input(myinput, debug=state["debug"])

        if route.get("use_tool"):
            answer = run_tool(myinput, route, messages, traces)
            print(answer)
            continue

        if not route.get("use_tool"):
            answer = chat_with_memory(messages, myinput)
            print(answer)
            continue


if __name__ == "__main__":
    main()
