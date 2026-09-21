from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    db_app_url: str = "http://app:8000"
    vector_app_url: str = "http://vector-app:8000"
    api_key: str = ""

    query_timeout_ms: int = Field(default=5000, ge=500, le=60000)
    max_search_limit: int = Field(default=100, ge=1, le=500)

    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
