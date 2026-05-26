import os
import subprocess
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
provider = os.getenv("LLM_PROVIDER", "codex_cli").strip().lower()

api_key = os.getenv("API_KEY")
base_url = os.getenv("LLM_BASE_URL")
model = os.getenv("LLM_MODEL")
client = None

codex_cli_path = os.getenv("CODEX_CLI_PATH", "codex.cmd").strip() or "codex.cmd"
codex_model = os.getenv("CODEX_MODEL", "").strip()
codex_timeout_seconds = int(os.getenv("CODEX_TIMEOUT_SECONDS", "120"))


def get_provider() -> str:
    return provider


def is_llm_configured() -> bool:
    if provider == "codex_cli":
        return True

    return bool(api_key and base_url and model)


def build_prompt_from_messages(messages: list) -> str:
    lines = [
        "You are the backend chat model for an AI job-search assistant.",
        "Reply in Chinese unless the user explicitly asks otherwise.",
        "Do not reveal chain-of-thought.",
        "Do not output <think> tags.",
        "Return only the final assistant reply.",
        "",
        "Conversation:",
    ]

    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        lines.append(f"[{role}]")
        lines.append(str(content).strip())
        lines.append("")

    lines.append("[ASSISTANT]")
    return "\n".join(lines).strip()


def run_codex_cli(messages: list) -> str:
    prompt = build_prompt_from_messages(messages)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as output_file:
        output_path = output_file.name

    command = [
        "cmd",
        "/c",
        codex_cli_path,
        "-a",
        "never",
        "-s",
        "read-only",
        "exec",
        "--skip-git-repo-check",
        "--color",
        "never",
        "--output-last-message",
        output_path,
        "-C",
        str(BASE_DIR),
        "-",
    ]

    if codex_model:
        command[3:3] = ["-m", codex_model]

    try:
        result = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=codex_timeout_seconds,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Codex CLI 不可用。请确认 codex 已安装，或在 .env 中配置 CODEX_CLI_PATH。"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Codex CLI 响应超时。") from exc

    try:
        output_text = Path(output_path).read_text(encoding="utf-8").strip()
    except OSError:
        output_text = ""
    finally:
        try:
            Path(output_path).unlink(missing_ok=True)
        except OSError:
            pass

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        raise RuntimeError(
            f"Codex CLI 调用失败：{stderr or stdout or f'exit code {result.returncode}'}"
        )

    if output_text:
        return output_text

    stdout = (result.stdout or "").strip()
    if stdout:
        return stdout

    raise RuntimeError("Codex CLI 未返回可用内容。")


def get_client():
    global client

    if provider != "openai_api":
        raise RuntimeError("当前 provider 不是 openai_api，无需创建 OpenAI client。")

    if not is_llm_configured():
        raise RuntimeError("LLM is not configured. Please set API_KEY, LLM_BASE_URL, and LLM_MODEL.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required to call the LLM.") from exc

    if client is None:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    return client


def call_llm(messages: list) -> str:
    response = create_chat_completion(messages=messages)
    return response["choices"][0]["message"].get("content") or ""


def create_chat_completion(
    messages: list,
    tools: list | None = None,
    tool_choice: str = "auto",
) -> dict:
    if provider == "codex_cli":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": run_codex_cli(messages),
                        "tool_calls": None,
                    }
                }
            ]
        }

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    response = get_client().chat.completions.create(**payload)

    if hasattr(response, "model_dump"):
        return response.model_dump()

    return {
        "choices": [
            {
                "message": {
                    "role": response.choices[0].message.role,
                    "content": response.choices[0].message.content,
                    "tool_calls": getattr(response.choices[0].message, "tool_calls", None),
                }
            }
        ]
    }


def complete_message(
    messages: list,
    tools: list | None = None,
    tool_choice: str = "auto",
) -> dict:
    response = create_chat_completion(
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
    )
    return response["choices"][0]["message"]
