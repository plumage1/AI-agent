import json

from core.redis_client import redis_client
from core.config import settings
from prompts.job_coach_prompt import SYSTEM_PROMPT

SESSION_TTL_SECONDS = settings.session_ttl_seconds

def build_session_key(session_id: str) -> str:
    return f"myagent:session:{session_id}"


def create_session() -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT}
        ],
        "traces": []
    }

def list_sessions() -> list[str]:
    keys = redis_client.keys("myagent:session:*")

    session_ids = []

    for key in keys:
        session_id = key.replace("myagent:session:", "")
        session_ids.append(session_id)

    return sorted(session_ids)

def get_session(session_id: str) -> dict:
    key = build_session_key(session_id)
    data = redis_client.get(key)

    if data is None:
        session = create_session()
        save_session(session_id, session)
        return session

    return json.loads(data)

def get_session_ttl(session_id: str) -> int:
    key = build_session_key(session_id)
    return redis_client.ttl(key)

def save_session(session_id: str, session: dict) -> None:
    key = build_session_key(session_id)
    redis_client.set(
        key,
        json.dumps(session, ensure_ascii=False),
        ex=SESSION_TTL_SECONDS
    )

def session_exists(session_id: str) -> bool:
    key = build_session_key(session_id)
    return redis_client.exists(key) == 1

def delete_session(session_id: str) -> bool:
    key = build_session_key(session_id)
    return redis_client.delete(key) == 1
