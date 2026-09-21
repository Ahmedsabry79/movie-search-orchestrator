from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Filter(StrictModel):
    field: str = Field(min_length=1, max_length=60)
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "is_null", "not_null"] = "eq"
    value: str | int | float | list[str | int | float] | None = None


class TextMatch(StrictModel):
    fields: list[Literal["title", "original_title", "tagline"]] = Field(default_factory=lambda: ["title", "original_title"], min_length=1, max_length=3)
    query: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def nonempty_query(self):
        if not self.query.strip():
            raise ValueError("Text query cannot be blank")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("Repeated text search field")
        return self


class Sort(StrictModel):
    field: str = Field(min_length=1, max_length=60)
    direction: Literal["asc", "desc"] = "desc"


class Metric(StrictModel):
    function: Literal["count", "sum", "avg", "min", "max"]
    field: str | None = None
    alias: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")


class SearchRequest(StrictModel):
    operation: Literal["list", "count", "aggregate"] = "list"
    filters: list[Filter] = Field(default_factory=list, max_length=20)
    text_match: TextMatch | None = None
    sort: list[Sort] = Field(default_factory=list, max_length=3)
    group_by: Literal["genre", "production_company", "release_year", "director", "actor", "original_language", "keyword", "production_country", "spoken_language"] | None = None
    metrics: list[Metric] = Field(default_factory=list, max_length=5)
    movie_ids: list[Annotated[int, Field(gt=0, lt=2**63)]] | None = Field(default=None, max_length=10000)
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=100000)

    @model_validator(mode="after")
    def operation_fields(self):
        if self.operation != "aggregate" and (self.group_by or self.metrics):
            raise ValueError("group_by and metrics require operation=aggregate")
        if self.operation == "count" and self.sort:
            raise ValueError("Sorting is not valid for a count")
        if self.movie_ids is not None and any(value <= 0 for value in self.movie_ids):
            raise ValueError("movie_ids must be positive")
        aliases = [metric.alias for metric in self.metrics]
        if len(set(aliases)) != len(aliases) or set(aliases) & {"group_id", "group_value"}:
            raise ValueError("Metric aliases must be unique and cannot use group_id or group_value")
        return self
