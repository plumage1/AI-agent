import logging
from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from core.config import settings

logger = logging.getLogger(__name__)
_CHECKPOINTER_STATUS = {
    "backend": "unknown",
    "detail": "",
}


def apply_redis_timeouts(redis_url: str) -> str:
    parsed = urlparse(redis_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.setdefault("socket_connect_timeout", str(settings.redis_socket_connect_timeout))
    query.setdefault("socket_timeout", str(settings.redis_socket_timeout))
    return urlunparse(parsed._replace(query=urlencode(query)))


def build_redis_url() -> str:
    if settings.redis_url:
        return apply_redis_timeouts(settings.redis_url)
    return apply_redis_timeouts(
        f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
    )


@lru_cache(maxsize=1)
def get_checkpointer():
    global _CHECKPOINTER_STATUS
    backend = (settings.langgraph_checkpointer_backend or "auto").strip().lower()

    if not settings.redis_enabled and backend in {"auto", "redis"}:
        backend = "memory" if backend == "auto" else backend

    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        _CHECKPOINTER_STATUS = {
            "backend": "memory",
            "detail": "Configured to use in-memory checkpointer.",
        }
        logger.info("Using LangGraph in-memory checkpointer.")
        return MemorySaver()

    if backend in {"auto", "redis"}:
        redis_url = build_redis_url()
        try:
            from langgraph.checkpoint.redis import RedisSaver

            saver = RedisSaver(redis_url=redis_url)
            saver.setup()
            _CHECKPOINTER_STATUS = {
                "backend": "redis",
                "detail": redis_url,
            }
            logger.info("Using LangGraph Redis checkpointer: %s", redis_url)
            return saver
        except Exception as exc:
            if backend == "redis":
                raise RuntimeError(
                    f"Redis checkpointer initialization failed: {exc}"
                ) from exc

            logger.warning(
                "Redis checkpointer unavailable, falling back to in-memory saver: %s",
                exc,
            )

    from langgraph.checkpoint.memory import MemorySaver

    _CHECKPOINTER_STATUS = {
        "backend": "memory",
        "detail": "Redis checkpointer unavailable; using in-memory fallback.",
    }
    logger.info("Falling back to LangGraph in-memory checkpointer.")
    return MemorySaver()


def get_thread_config(session_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": session_id,
        }
    }


def get_checkpointer_status() -> dict:
    get_checkpointer()
    return dict(_CHECKPOINTER_STATUS)
