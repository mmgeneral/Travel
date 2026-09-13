"""Configuration for the user service.

Every variable is read from the environment with the ``USER_SERVICE_`` prefix,
so it can never collide with the historical research code at the repo root.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Only ever read ``user_service/.env`` – never the repository root ``.env``.
_ENV_FILE = Path(__file__).resolve().parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="USER_SERVICE_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "user-service"

    # Sync SQLAlchemy URL (psycopg3 driver).
    database_url: str = (
        "postgresql+psycopg://user_service:user_service@localhost:5433/user_service"
    )

    supabase_url: str = ""          # e.g. https://xxxx.supabase.co
    supabase_jwt_secret: str = ""   # legacy HS256 projects only
    supabase_jwt_audience: str = "authenticated"

    # MUST stay 0 in production. Enables ``Authorization: Bearer dev:<id>``.
    dev_mode: bool = False

    # Comma separated list of allowed browser origins. Empty = CORS disabled.
    cors_origins: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def supabase_base_url(self) -> str:
        return self.supabase_url.strip().rstrip("/")

    @property
    def jwks_url(self) -> str:
        return f"{self.supabase_base_url}/auth/v1/.well-known/jwks.json"

    @property
    def issuer(self) -> str | None:
        base = self.supabase_base_url
        return f"{base}/auth/v1" if base else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
