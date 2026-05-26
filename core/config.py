import os
from dataclasses import dataclass

from dotenv import load_dotenv


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


settings = Settings(
    cors_allow_origins=parse_csv(os.getenv("CORS_ALLOW_ORIGINS"))
)
