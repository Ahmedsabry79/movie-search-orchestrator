from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    api_key: str = ""

    postgres_host: str = "db"
    postgres_port: int = 5432
    postgres_db: str = "movies"
    postgres_user: str = "movies"
    postgres_password: str = "movies_local"

    milvus_uri: str = "http://milvus:19530"
    milvus_collection: str = "movies_hybrid_v2"
    embedding_dimension: int = 1024

    model_base_url: str = "http://model-service:8000/v1"
    model_api_key: str = "movie-agent-local"
    embedding_model: str = "BAAI/bge-m3"

    index_batch_size: int = Field(default=64, ge=1, le=512)
    search_candidate_multiplier: int = Field(default=4, ge=1, le=20)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    auto_index_on_start: bool = True
    index_if_empty_only: bool = True

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
