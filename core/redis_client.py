import os
import redis
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False


load_dotenv()


def redis_enabled() -> bool:
    return os.getenv("REDIS_ENABLED", "true").strip().lower() not in {"0", "false", "no"}


class DisabledRedisClient:
    def __getattr__(self, _name):
        raise redis.exceptions.ConnectionError("Redis is disabled for this process.")


if redis_enabled():
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        decode_responses=True,
        socket_connect_timeout=float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "1")),
        socket_timeout=float(os.getenv("REDIS_SOCKET_TIMEOUT", "1")),
    )
else:
    redis_client = DisabledRedisClient()
