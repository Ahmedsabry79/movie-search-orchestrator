from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    postgres_host: str = "db"
    postgres_port: int = 5432
    postgres_db: str = "movies"
    postgres_user: str = "movies"
    postgres_password: str = "movies_local"

    vector_app_url: str = "http://vector-app:8000"
    api_key: str = ""

    fuzzy_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    query_timeout_ms: int = Field(default=5000, ge=500, le=60000)
    max_search_limit: int = Field(default=100, ge=1, le=500)

    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8000

    @property
    def postgres_dsn(self) -> str:
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"dbname={self.postgres_db} user={self.postgres_user} "
            f"password={self.postgres_password}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
