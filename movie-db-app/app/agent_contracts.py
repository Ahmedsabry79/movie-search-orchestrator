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


class MovieFilterCriteria(StrictModel):
    """Structured SQL-searchable movie criteria.

    Overview is intentionally absent. Plot/overview meaning must be handled by semantic
    retrieval through the vector service.
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

    # cast.character is fuzzy. Crew department/job remain literal containment filters.
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
    fuzzy_threshold: float | None = Field(default=None, gt=0.0, le=1.0)

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

    def has_filters(self) -> bool:
        ignored = {"fuzzy_threshold"}
        data = self.model_dump()
        return any(
            data.get(name) not in (None, [], "")
            for name in MovieFilterCriteria.model_fields
            if name not in ignored
        )


class SearchSort(StrictModel):
    field: Literal[
        "title", "release_date", "release_year", "budget", "revenue", "runtime",
        "vote_average", "vote_count", "popularity", "fuzzy_score"
    ]
    direction: Literal["asc", "desc"] = "desc"


class AgentMovieSearchRequest(MovieFilterCriteria):
    """High-level movie retrieval criteria used by the agent/MCP integration."""

    sort: list[SearchSort] = Field(default_factory=list, max_length=3)
    limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def require_filter(self) -> "AgentMovieSearchRequest":
        if not self.has_filters():
            raise ValueError("At least one structured search criterion is required")
        return self


class FuzzyMatchEvidence(StrictModel):
    field: str
    query: str
    matched_value: str
    similarity: float = Field(ge=0.0, le=1.0)


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
    # Backward-compatible title/original-title/tagline ranking score.
    fuzzy_score: float | None = None
    # Per fuzzy-capable criterion evidence. The agent should use this when judging match quality.
    fuzzy_matches: list[FuzzyMatchEvidence] = Field(default_factory=list)
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


AnalyticsDimension = Literal[
    "genre",
    "keyword",
    "production_company",
    "production_country",
    "spoken_language",
    "cast_member",
    "crew_member",
    "director",
    "cast_character",
    "crew_department",
    "crew_job",
    "original_language",
    "status",
    "release_year",
]
AnalyticsMetricField = Literal[
    "budget",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count",
    "popularity",
]


class AnalyticsMetric(StrictModel):
    function: Literal["count", "sum", "avg", "min", "max"]
    field: AnalyticsMetricField | None = None
    alias: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")

    @model_validator(mode="after")
    def validate_metric(self) -> "AnalyticsMetric":
        if self.function == "count":
            if self.field is not None:
                raise ValueError("count always counts distinct movies; omit field")
        elif self.field is None:
            raise ValueError(f"{self.function} requires a numeric movie field")
        return self


class AnalyticsSort(StrictModel):
    field: str = Field(min_length=1, max_length=60)
    direction: Literal["asc", "desc"] = "desc"


class AgentAnalyticsRequest(StrictModel):
    """Controlled catalog analytics with automatic joins owned by movie-db-app.

    `group_by` may contain one or more logical dimensions. The DB app creates only
    allow-listed joins; the caller never supplies SQL/table names/join expressions.
    """

    filters: MovieFilterCriteria = Field(default_factory=MovieFilterCriteria)
    group_by: list[AnalyticsDimension] = Field(default_factory=list, max_length=3)
    metrics: list[AnalyticsMetric] = Field(
        default_factory=lambda: [AnalyticsMetric(function="count", alias="movie_count")],
        min_length=1,
        max_length=6,
    )
    sort: list[AnalyticsSort] = Field(default_factory=list, max_length=4)
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=100_000)

    @model_validator(mode="after")
    def validate_request(self) -> "AgentAnalyticsRequest":
        if len(set(self.group_by)) != len(self.group_by):
            raise ValueError("group_by dimensions must be unique")
        aliases = [metric.alias for metric in self.metrics]
        if len(set(aliases)) != len(aliases):
            raise ValueError("metric aliases must be unique")
        reserved = set(self.group_by) | {f"{dimension}_id" for dimension in self.group_by}
        if set(aliases) & reserved:
            raise ValueError("metric aliases cannot conflict with group-by output fields")
        return self


class AgentAnalyticsResponse(StrictModel):
    status: Literal["ok", "no_matches"]
    matched_movie_count: int
    group_by: list[AnalyticsDimension]
    metrics: list[dict]
    total_groups: int
    returned_count: int
    has_more: bool
    filters: dict
    results: list[dict]


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
