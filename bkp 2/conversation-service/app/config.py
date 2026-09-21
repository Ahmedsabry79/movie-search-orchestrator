from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    api_key: str = ""

    db_app_url: str = "http://app:8000"
    mcp_url: str = "http://mcp-server:8000/mcp"

    model_base_url: str = "http://model-service:8000/v1"
    model_api_key: str = "movie-agent-local"
    generator_model: str = "movie-agent-generator"

    planning_reasoning_effort: str = "medium"
    agent_reasoning_effort: str = "medium"
    planner_max_tokens: int = 1200
    agent_max_tokens: int = 2400

    max_agent_iterations: int = 8
    max_tool_calls: int = 10
    max_tool_failures: int = 3
    mcp_transport_retries: int = 3
    mcp_retry_base_seconds: float = 0.5

    conversation_history_limit: int = 20
    schema_cache_ttl_seconds: int = 900
    model_timeout_seconds: float = 180.0
    db_timeout_seconds: float = 15.0


@lru_cache

def get_settings() -> Settings:
    return Settings()
