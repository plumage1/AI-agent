from core.llm import call_llm

def chat_with_memory(messages: list, user_input: str) -> str:
    messages.append({"role": "user", "content": user_input})

    answer = call_llm(messages)

    messages.append({"role": "assistant", "content": answer})

    return answer