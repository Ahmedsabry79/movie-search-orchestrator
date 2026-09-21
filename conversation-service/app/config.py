from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
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
    recovery_agent_max_tokens: int = 1600

    max_agent_iterations: int = 8
    max_tool_calls: int = 10
    max_tool_failures: int = 3
    mcp_transport_retries: int = 3
    mcp_retry_base_seconds: float = 0.5

    # A "turn" means one user request plus its assistant response. The current
    # user request is included in this budget, so 3 turns typically means:
    # previous user+assistant, previous user+assistant, current user.
    conversation_history_turns: int = 3
    # On a context-window recovery, keep only one completed previous turn plus
    # the current user request and restart the complete planning/tool cycle.
    recovery_previous_turns: int = 1

    # Long-term conversation memory. The full transcript remains persisted, while
    # older turns that fall outside the active window are summarized incrementally.
    memory_compaction_enabled: bool = True
    memory_compaction_min_messages: int = Field(default=2, ge=1, le=50)
    memory_compaction_max_tokens: int = Field(default=1000, ge=128, le=4000)
    memory_compaction_message_limit: int = Field(default=200, ge=10, le=500)
    memory_compaction_max_chars: int = Field(default=16000, ge=2000, le=100000)

    # Retrieval quality thresholds used by the orchestrator when deciding whether
    # literal/fuzzy retrieval is convincing enough or semantic fallback is warranted.
    fuzzy_accept_similarity: float = Field(default=0.50, ge=0.0, le=1.0)
    fuzzy_strong_similarity: float = Field(default=0.75, ge=0.0, le=1.0)

    schema_cache_ttl_seconds: int = 900
    model_timeout_seconds: float = 180.0
    db_timeout_seconds: float = 15.0

    @model_validator(mode="after")
    def validate_thresholds(self) -> "Settings":
        if self.fuzzy_strong_similarity < self.fuzzy_accept_similarity:
            raise ValueError("FUZZY_STRONG_SIMILARITY must be >= FUZZY_ACCEPT_SIMILARITY")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
