import json

from agents.langgraph_runtime import clear_thread_state, get_thread_state, update_thread_state
from core.config import settings
from core.redis_client import redis_client
from prompts.job_coach_prompt import SYSTEM_PROMPT

SESSION_TTL_SECONDS = settings.session_ttl_seconds
MEMORY_SESSIONS = {}


def build_session_key(session_id: str) -> str:
    return f"myagent:session:{session_id}"


def create_session() -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT}
        ],
        "traces": [],
    }


def list_sessions() -> list[str]:
    try:
        keys = redis_client.keys("myagent:session:*")
        session_ids = []

        for key in keys:
            session_id = key.replace("myagent:session:", "")
            session_ids.append(session_id)

        return sorted(session_ids)
    except Exception:
        return sorted(MEMORY_SESSIONS.keys())


def get_session(session_id: str) -> dict:
    key = build_session_key(session_id)
    data = None
    try:
        data = redis_client.get(key)
    except Exception:
        data = MEMORY_SESSIONS.get(session_id)
    stored_session = json.loads(data) if data else create_session()
    graph_state = get_thread_state(session_id)

    if graph_state:
        stored_session["messages"] = graph_state.get("messages", stored_session.get("messages", []))
        stored_session["traces"] = graph_state.get("traces", stored_session.get("traces", []))
        if "interview" in graph_state and graph_state.get("interview") is not None:
            stored_session["interview"] = graph_state.get("interview")

    if not stored_session.get("messages"):
        stored_session["messages"] = create_session()["messages"]
    if "traces" not in stored_session:
        stored_session["traces"] = []

    if data is None:
        save_session(session_id, stored_session)

    return stored_session


def get_session_ttl(session_id: str) -> int:
    key = build_session_key(session_id)
    try:
        return redis_client.ttl(key)
    except Exception:
        return SESSION_TTL_SECONDS if session_id in MEMORY_SESSIONS else -2


def save_session(session_id: str, session: dict) -> None:
    key = build_session_key(session_id)
    payload = json.dumps(session, ensure_ascii=False)
    MEMORY_SESSIONS[session_id] = payload

    try:
        update_thread_state(session_id, {"interview": session.get("interview")})
    except Exception:
        pass

    try:
        redis_client.set(
            key,
            payload,
            ex=SESSION_TTL_SECONDS,
        )
    except Exception:
        return


def session_exists(session_id: str) -> bool:
    key = build_session_key(session_id)
    try:
        if redis_client.exists(key) == 1:
            return True
    except Exception:
        if session_id in MEMORY_SESSIONS:
            return True

    graph_state = get_thread_state(session_id)
    return bool(graph_state)


def delete_session(session_id: str) -> bool:
    key = build_session_key(session_id)
    had_graph_state = bool(get_thread_state(session_id))
    deleted = False
    try:
        deleted = redis_client.delete(key) == 1
    except Exception:
        deleted = False

    if session_id in MEMORY_SESSIONS:
        MEMORY_SESSIONS.pop(session_id, None)
        deleted = True

    clear_thread_state(session_id)
    return deleted or had_graph_state


def reset_chat_session(session_id: str) -> dict:
    clear_thread_state(session_id)
    session = create_session()
    save_session(session_id, session)
    return session
