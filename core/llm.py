import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("LLM_BASE_URL")
model = os.getenv("LLM_MODEL")

if not api_key:
    raise ValueError("Missing API_KEY in .env")

if not base_url:
    raise ValueError("Missing LLM_BASE_URL in .env")

if not model:
    raise ValueError("Missing LLM_MODEL in .env")


client = OpenAI(
    api_key=api_key,
    base_url=base_url
)


def call_llm(messages: list) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False
    )

    return response.choices[0].message.content