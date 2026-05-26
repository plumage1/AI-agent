import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False


load_dotenv()


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


@dataclass
class Settings:
    app_name: str = os.getenv("APP_NAME", "AI Job Agent")
    app_env: str = os.getenv("APP_ENV", "local")
    admin_token: str | None = os.getenv("ADMIN_TOKEN")
    cors_allow_origins: list[str] = None
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
    max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "20000"))
    max_top_k: int = int(os.getenv("MAX_TOP_K", "10"))
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", str(60 * 60 * 24)))
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_url: str | None = os.getenv("REDIS_URL")
    langgraph_checkpointer_backend: str = os.getenv("LANGGRAPH_CHECKPOINTER_BACKEND", "auto")


settings = Settings(
    cors_allow_origins=parse_csv(os.getenv("CORS_ALLOW_ORIGINS"))
)
