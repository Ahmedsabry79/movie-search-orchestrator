from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


SearchMode = Literal["hybrid", "dense", "bm25"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=10, ge=1, le=100)
    mode: SearchMode = "hybrid"
    year_from: int | None = Field(default=None, ge=1870, le=2200)
    year_to: int | None = Field(default=None, ge=1870, le=2200)
    min_vote_average: float | None = Field(default=None, ge=0.0, le=10.0)
    genre: str | None = Field(default=None, max_length=100)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("query must not be blank")
        return value


class SearchHit(BaseModel):
    movie_id: int
    score: float
    normalized_score: float | None = None
    quality: Literal["strong", "acceptable", "weak", "unscored"] = "unscored"
    title: str
    original_title: str = ""
    tagline: str = ""
    overview: str = ""
    genres: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    release_year: int | None = None
    vote_average: float | None = None
    popularity: float | None = None


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    count: int
    accepted_count: int = 0
    top_quality: Literal["strong", "acceptable", "weak", "unscored", "no_match"] = "no_match"
    thresholds: dict[str, float] = Field(default_factory=dict)
    results: list[SearchHit]


class ReindexResponse(BaseModel):
    indexed: int
    collection: str


class IndexStatus(BaseModel):
    collection: str
    exists: bool
    rows: int | None = None
