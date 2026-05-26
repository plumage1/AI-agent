from agents.langgraph_runtime import clear_thread_state, run_langgraph_agent_turn, show_traces
from prompts.job_coach_prompt import SYSTEM_PROMPT

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

        if command == "exit":
            break

        if command == "clear":
            clear_thread_state("cli")
            messages.clear()
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
            traces.clear()
            print("Memory cleared.")
            continue

        if command == "trace":
            show_traces(traces)
            continue

        if command == "debug":
            state["debug"] = not state["debug"]
            print(f"Debug mode: {state['debug']}")
            continue

        if command == "help":
            print("""
Available commands:
- exit: quit the program
- clear: clear conversation memory
- trace: show tool execution traces
- debug: toggle debug output
- help: show this help message
""")
            continue

        result = run_langgraph_agent_turn(myinput, messages, traces, session_id="cli")
        print(result["answer"])


if __name__ == "__main__":
    main()
