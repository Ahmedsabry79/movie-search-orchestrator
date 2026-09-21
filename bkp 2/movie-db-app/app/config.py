import os
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import URL


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    db_schema: str = "movie_catalog"
    dataset_version: str = "tmdb-5000-preprocessing-v1"
    fuzzy_threshold: float = float(os.getenv("FUZZY_THRESHOLD", "0.30"))
    fuzzy_min_gap: float = float(os.getenv("FUZZY_MIN_GAP", "0.10"))
    query_timeout_ms: int = int(os.getenv("QUERY_TIMEOUT_MS", "5000"))
    api_key: str | None = os.getenv("API_KEY") or None
    download_url: str = os.getenv("DATASET_URL", "https://www.kaggle.com/api/v1/datasets/download/tmdb/tmdb-movie-metadata")

    @property
    def database_url(self):
        return os.getenv("DATABASE_URL") or URL.create(
            "postgresql+psycopg", username=os.getenv("POSTGRES_USER", "movies"),
            password=os.getenv("POSTGRES_PASSWORD", "movies_local"),
            host=os.getenv("POSTGRES_HOST", "localhost"), port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "movies"),
        )


settings = Settings()
