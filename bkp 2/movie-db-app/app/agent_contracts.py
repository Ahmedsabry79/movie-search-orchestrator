from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


MatchMode = Literal["all", "any"]
ReferenceType = Literal[
    "genre",
    "keyword",
    "production_company",
    "production_country",
    "spoken_language",
    "cast",
    "crew",
    "person",
    "status",
    "original_language",
]


class StringSetFilter(StrictModel):
    values: list[str] = Field(min_length=1, max_length=20)
    match: MatchMode = "all"

    @field_validator("values")
    @classmethod
    def clean_values(cls, values: list[str]) -> list[str]:
        cleaned = [" ".join(value.split()) for value in values if value and value.strip()]
        if not cleaned:
            raise ValueError("values cannot be empty")
        return list(dict.fromkeys(cleaned))


class NumericRange(StrictModel):
    eq: float | None = None
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def validate_range(self) -> "NumericRange":
        if self.eq is None and self.min is None and self.max is None:
            raise ValueError("At least one of eq/min/max must be supplied")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min cannot be greater than max")
        return self


class IntegerRange(StrictModel):
    eq: int | None = None
    min: int | None = None
    max: int | None = None

    @model_validator(mode="after")
    def validate_range(self) -> "IntegerRange":
        if self.eq is None and self.min is None and self.max is None:
            raise ValueError("At least one of eq/min/max must be supplied")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min cannot be greater than max")
        return self


class SearchSort(StrictModel):
    field: Literal[
        "title", "release_date", "release_year", "budget", "revenue", "runtime",
        "vote_average", "vote_count", "popularity", "fuzzy_score"
    ]
    direction: Literal["asc", "desc"] = "desc"


class DateRange(StrictModel):
    eq: date | None = None
    from_date: date | None = None
    to_date: date | None = None

    @model_validator(mode="after")
    def validate_range(self) -> "DateRange":
        if self.eq is None and self.from_date is None and self.to_date is None:
            raise ValueError("At least one of eq/from_date/to_date must be supplied")
        if self.from_date is not None and self.to_date is not None and self.from_date > self.to_date:
            raise ValueError("from_date cannot be after to_date")
        return self


class AgentMovieSearchRequest(StrictModel):
    """High-level movie criteria used by the agent/MCP integration.

    Criteria across fields are ANDed. Multiple values inside a StringSetFilter use
    its explicit `match` mode. Title/original-title/tagline and person-name criteria
    are fuzzy matched by the DB service itself.
    """

    movie_ids: list[int] = Field(default_factory=list, max_length=50)

    titles: list[str] = Field(default_factory=list, max_length=10)
    original_titles: list[str] = Field(default_factory=list, max_length=10)
    taglines: list[str] = Field(default_factory=list, max_length=10)

    genres: StringSetFilter | None = None
    keywords: StringSetFilter | None = None
    production_companies: StringSetFilter | None = None
    production_countries: StringSetFilter | None = None
    spoken_languages: StringSetFilter | None = None
    cast: StringSetFilter | None = None
    crew: StringSetFilter | None = None

    cast_characters: StringSetFilter | None = None
    crew_departments: StringSetFilter | None = None
    crew_jobs: StringSetFilter | None = None

    budget: IntegerRange | None = None
    original_languages: StringSetFilter | None = None
    popularity: NumericRange | None = None
    release_date: DateRange | None = None
    release_year: IntegerRange | None = None
    revenue: IntegerRange | None = None
    runtime: NumericRange | None = None
    statuses: StringSetFilter | None = None
    vote_average: NumericRange | None = None
    vote_count: IntegerRange | None = None

    homepage_contains: str | None = Field(default=None, max_length=500)
    overview_contains: str | None = Field(default=None, max_length=1000)

    sort: list[SearchSort] = Field(default_factory=list, max_length=3)

    fuzzy_threshold: float | None = Field(default=None, gt=0.0, le=1.0)
    limit: int = Field(default=20, ge=1, le=100)

    @field_validator("movie_ids")
    @classmethod
    def positive_movie_ids(cls, values: list[int]) -> list[int]:
        if any(value <= 0 or value >= 2**63 for value in values):
            raise ValueError("movie_ids must contain positive 64-bit integers")
        return list(dict.fromkeys(values))

    @field_validator("titles", "original_titles", "taglines")
    @classmethod
    def clean_fuzzy_lists(cls, values: list[str]) -> list[str]:
        cleaned = [" ".join(value.split()) for value in values if value and value.strip()]
        return list(dict.fromkeys(cleaned))

    @model_validator(mode="after")
    def require_filter(self) -> "AgentMovieSearchRequest":
        ignored = {"limit", "fuzzy_threshold", "sort"}
        supplied = [
            name
            for name, value in self.model_dump().items()
            if name not in ignored and value not in (None, [], "")
        ]
        if not supplied:
            raise ValueError("At least one structured search criterion is required")
        return self


class AgentMovieSearchHit(StrictModel):
    movie_id: int
    title: str
    original_title: str | None = None
    tagline: str | None = None
    release_date: date | None = None
    status: str | None = None
    original_language: str | None = None
    runtime: float | None = None
    budget: int | None = None
    revenue: int | None = None
    popularity: float | None = None
    vote_average: float | None = None
    vote_count: int | None = None
    fuzzy_score: float | None = None
    genres: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    production_companies: list[str] = Field(default_factory=list)
    production_countries: list[str] = Field(default_factory=list)
    spoken_languages: list[str] = Field(default_factory=list)
    cast: list[str] = Field(default_factory=list)
    crew: list[str] = Field(default_factory=list)


class AgentMovieSearchResponse(StrictModel):
    status: Literal["ok", "no_matches"]
    count: int
    returned_count: int
    has_more: bool = False
    criteria: dict
    semantic_fallback_recommended: bool = False
    results: list[AgentMovieSearchHit]


class ReferenceResolveRequest(StrictModel):
    reference_type: ReferenceType
    query: str | None = Field(default=None, max_length=300)
    limit: int = Field(default=20, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        return value or None


class ReferenceMatch(StrictModel):
    value: str
    code: str | None = None
    id: int | None = None
    score: float = 1.0
    extra: dict = Field(default_factory=dict)


class ReferenceResolutionResponse(StrictModel):
    reference_type: ReferenceType
    query: str | None = None
    count: int
    results: list[ReferenceMatch]


class VectorIndexMovie(StrictModel):
    movie_id: int
    title: str
    original_title: str = ""
    tagline: str = ""
    overview: str = ""
    genres: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    release_year: int | None = None
    vote_average: float | None = None
    popularity: float | None = None


class VectorIndexPage(StrictModel):
    total: int
    offset: int
    limit: int
    next_offset: int | None = None
    items: list[VectorIndexMovie]
