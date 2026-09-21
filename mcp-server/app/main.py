from __future__ import annotations

from typing import Annotated, Literal

import asyncio
import httpx
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import get_settings
from .schemas import (
    DatabaseAnalyticsRequest,
    DatabaseAnalyticsResponse,
    DatabaseSearchCriteria,
    DatabaseSearchResponse,
    DateRange,
    IntegerRange,
    MovieFilterCriteria,
    NumericRange,
    ReferenceResolutionResponse,
    ReferenceType,
    SearchSort,
    StringSetFilter,
    SemanticSearchRequest,
    SemanticSearchResponse,
)


settings = get_settings()

MCP_INSTRUCTIONS = """
Movie-domain retrieval and analytics tools for an LLM agent.

Service ownership is strict:
- PostgreSQL is accessed only through movie-db-app.
- Milvus is accessed only through movie-vector-app.
- This MCP server validates tool inputs and calls those service APIs; it contains no SQL
  and no direct Milvus client logic.

Routing policy:
1. FIND/LIST movies with explicit DB criteria -> search_movies_db.
   Fuzzy-capable inputs are title, original title, literal tagline, named relations,
   cast/crew person names, and cast.character. Search results return per-field
   fuzzy_matches with the query, actual DB value, and similarity so the agent can judge quality.
2. COUNT movies matching criteria without grouping -> count_movies_db.
3. Dataset-level questions requiring GROUP BY, JOIN-backed dimensions, COUNT/SUM/AVG/MIN/MAX,
   distributions or breakdowns -> analyze_movies_db. Joins are selected by logical dimensions;
   never construct SQL yourself. Multiple group_by dimensions are supported.
4. Use resolve_reference_value when a canonical DB value should be checked or normalized first,
   e.g. genre, language, country, company, person or status.
5. Overview is intentionally NOT available to structured SQL search. Any plot/overview/theme
   meaning, or paraphrased tagline meaning, must use search_movies_semantic.
6. If structured/fuzzy movie retrieval returns no convincing result, use search_movies_semantic as
   fallback when a meaningful semantic query can be formed. Hybrid/dense semantic results expose
   normalized score + quality; weak results are candidates only, not confirmed matches.
7. After identifying candidate IDs, use get_movie_details when complete metadata/relationship
   evidence is needed before answering.
8. get_database_schema exposes the authoritative tables, columns, relationships, searchable
   structured fields, fuzzy semantics and analytics contract. The conversation service loads it
   together with the complete MCP tool catalog at startup.
""".strip()

mcp = MCPServer("movie-agent-tools", instructions=MCP_INSTRUCTIONS)


def _headers() -> dict[str, str]:
    return {"X-API-Key": settings.api_key} if settings.api_key else {}


def _timeout() -> httpx.Timeout:
    seconds = max(settings.query_timeout_ms / 1000.0, 5.0)
    return httpx.Timeout(seconds, connect=5.0)


def _normalize_search_sort(value) -> list[SearchSort]:
    """Accept canonical sort objects plus compact LLM-friendly shorthand.

    Supported examples:
    - {"field": "vote_average", "direction": "desc"}
    - "-vote_average"  -> descending
    - "+release_year"  -> ascending
    - "vote_average:desc"
    - ["-vote_average", "-vote_count"]
    """
    if value in (None, [], ""):
        return []
    items = value if isinstance(value, list) else [value]
    normalized: list[SearchSort] = []
    for item in items:
        if isinstance(item, SearchSort):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            normalized.append(SearchSort.model_validate(item))
            continue
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Unsupported sort value: {item!r}")

        raw = item.strip()
        direction = "desc"
        if raw.startswith("-"):
            raw = raw[1:].strip()
            direction = "desc"
        elif raw.startswith("+"):
            raw = raw[1:].strip()
            direction = "asc"
        elif ":" in raw:
            raw, direction = (part.strip() for part in raw.split(":", 1))
        else:
            parts = raw.rsplit(None, 1)
            if len(parts) == 2 and parts[1].lower() in {"asc", "desc"}:
                raw, direction = parts[0].strip(), parts[1].lower()

        direction = direction.lower()
        if direction not in {"asc", "desc"}:
            raise ValueError(f"Sort direction must be asc/desc, got {direction!r}")
        normalized.append(SearchSort(field=raw, direction=direction))
    return normalized


async def _request_json(
    method: str,
    url: str,
    *,
    service_name: str,
    json: dict | None = None,
    params: dict | None = None,
) -> dict:
    try:
        async with httpx.AsyncClient(timeout=_timeout(), headers=_headers()) as client:
            response = await client.request(method, url, json=json, params=params)
        if response.status_code >= 400:
            detail = response.text[:1000]
            raise ToolError(f"{service_name} returned HTTP {response.status_code}: {detail}")
        return response.json()
    except ToolError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise ToolError(f"{service_name} request failed: {exc}") from exc


@mcp.tool()
async def get_database_schema() -> dict:
    """Fetch the authoritative movie database schema and search contracts from movie-db-app.

    This is primarily a host/context tool. It describes every movie-domain table, column,
    relation, supported search field and matching semantic. Normal movie requests should use
    the search/reference/detail tools instead of repeatedly calling this schema tool.
    """
    return await _request_json(
        "GET",
        f"{settings.db_app_url.rstrip('/')}/schemas",
        service_name="movie-db-app",
    )


@mcp.tool()
async def search_movies_db(
    movie_ids: list[int] | None = None,
    titles: list[str] | None = None,
    original_titles: list[str] | None = None,
    taglines: list[str] | None = None,
    genres: StringSetFilter | None = None,
    keywords: StringSetFilter | None = None,
    production_companies: StringSetFilter | None = None,
    production_countries: StringSetFilter | None = None,
    spoken_languages: StringSetFilter | None = None,
    cast: StringSetFilter | None = None,
    crew: StringSetFilter | None = None,
    cast_characters: StringSetFilter | None = None,
    crew_departments: StringSetFilter | None = None,
    crew_jobs: StringSetFilter | None = None,
    budget: IntegerRange | None = None,
    original_languages: StringSetFilter | None = None,
    popularity: NumericRange | None = None,
    release_date: DateRange | None = None,
    release_year: IntegerRange | None = None,
    revenue: IntegerRange | None = None,
    runtime: NumericRange | None = None,
    statuses: StringSetFilter | None = None,
    vote_average: NumericRange | None = None,
    vote_count: IntegerRange | None = None,
    homepage_contains: str | None = None,
    fuzzy_threshold: float | None = None,
    sort: str | SearchSort | list[str | SearchSort] | None = None,
    limit: int = 20,
) -> DatabaseSearchResponse:
    """Search movie-db-app using one or many structured/fuzzy criteria.

    Arguments are FLAT: pass movie criteria directly at the top level; do not wrap
    them inside a `criteria` object. Examples:
    - title typo: {"titles": ["Interstelar"]}
    - actor + year + rating: {"cast": {"values": ["Tom Hanks"]},
      "release_year": {"min": 2000}, "vote_average": {"min": 7.0}}
    - character fuzzy match: {"cast_characters": {"values": ["Coopr"]}}

    Criteria across fields are combined. Relation filters support `match=all|any`.
    Fuzzy matching is automatic for title, original title, literal tagline, named
    relations, cast/crew person names and cast.character. `sort` accepts canonical
    objects such as {"field":"vote_average","direction":"desc"} and safe
    shorthand such as "-vote_average". Returned hits include
    per-field `fuzzy_matches` similarity evidence.

    Overview/plot meaning is intentionally excluded; use search_movies_semantic.
    """
    try:
        criteria = DatabaseSearchCriteria(
            movie_ids=movie_ids or [],
            titles=titles or [],
            original_titles=original_titles or [],
            taglines=taglines or [],
            genres=genres,
            keywords=keywords,
            production_companies=production_companies,
            production_countries=production_countries,
            spoken_languages=spoken_languages,
            cast=cast,
            crew=crew,
            cast_characters=cast_characters,
            crew_departments=crew_departments,
            crew_jobs=crew_jobs,
            budget=budget,
            original_languages=original_languages,
            popularity=popularity,
            release_date=release_date,
            release_year=release_year,
            revenue=revenue,
            runtime=runtime,
            statuses=statuses,
            vote_average=vote_average,
            vote_count=vote_count,
            homepage_contains=homepage_contains,
            fuzzy_threshold=fuzzy_threshold,
            sort=_normalize_search_sort(sort),
            limit=min(limit, settings.max_search_limit),
        )
    except ValueError as exc:
        raise ToolError(f"Invalid structured movie search arguments: {exc}") from exc

    payload = criteria.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    data = await _request_json(
        "POST",
        f"{settings.db_app_url.rstrip('/')}/agent/search",
        service_name="movie-db-app",
        json=payload,
    )
    try:
        return DatabaseSearchResponse.model_validate(data)
    except ValueError as exc:
        raise ToolError(f"movie-db-app returned an invalid search response: {exc}") from exc


@mcp.tool()
async def count_movies_db(filters: MovieFilterCriteria | None = None) -> dict:
    """Count distinct movies matching optional structured criteria through movie-db-app.

    Use this for questions such as "how many movies have rating above 8?" when no grouped
    breakdown is requested. For "how many in each genre/year/country?", use analyze_movies_db.
    The same structured/fuzzy filter semantics as search_movies_db apply. Overview is not a
    structured filter.
    """
    payload = {
        "filters": (filters or MovieFilterCriteria()).model_dump(
            mode="json", exclude_none=True, exclude_defaults=True
        ),
        "group_by": [],
        "metrics": [{"function": "count", "alias": "movie_count"}],
        "limit": 1,
        "offset": 0,
    }
    data = await _request_json(
        "POST",
        f"{settings.db_app_url.rstrip('/')}/agent/analyze",
        service_name="movie-db-app",
        json=payload,
    )
    try:
        response = DatabaseAnalyticsResponse.model_validate(data)
    except ValueError as exc:
        raise ToolError(f"movie-db-app returned an invalid count response: {exc}") from exc
    count = 0
    if response.results:
        count = int(response.results[0].get("movie_count") or 0)
    return {
        "status": response.status,
        "count": count,
        "filters": response.filters,
    }


@mcp.tool()
async def analyze_movies_db(request: DatabaseAnalyticsRequest) -> DatabaseAnalyticsResponse:
    """Run controlled movie-catalog analytics with automatic allow-listed table joins.

    Use for grouped counts, distributions and numeric aggregates. `group_by` accepts up to
    three logical dimensions such as genre, keyword, production_company/country,
    spoken_language, cast_member, crew_member, director, cast_character, crew_department,
    crew_job, original_language, status, or release_year. Metrics support distinct movie
    count plus sum/avg/min/max over budget, revenue, runtime, vote_average, vote_count and
    popularity. Structured filters can be combined with the aggregation.

    Examples:
    - movies per genre: group_by=["genre"], metric count
    - average rating by genre and year: group_by=["genre","release_year"], avg vote_average
    - revenue by production company after 2010: production_company dimension + year filter
    - movie counts by cast character: group_by=["cast_character"], metric count

    The tool never accepts SQL, table names or join expressions; movie-db-app owns the joins.
    """
    request.limit = min(request.limit, 200)
    payload = request.model_dump(mode="json", exclude_none=True)
    data = await _request_json(
        "POST",
        f"{settings.db_app_url.rstrip('/')}/agent/analyze",
        service_name="movie-db-app",
        json=payload,
    )
    try:
        return DatabaseAnalyticsResponse.model_validate(data)
    except ValueError as exc:
        raise ToolError(f"movie-db-app returned an invalid analytics response: {exc}") from exc


@mcp.tool()
async def search_movies_semantic(
    query: Annotated[str, Field(min_length=1, max_length=4000, description="Natural-language semantic movie query")],
    limit: Annotated[int, Field(ge=1, le=100, description="Maximum number of candidates")] = 10,
    mode: Annotated[Literal["hybrid", "dense", "bm25"], Field(description="Retrieval mode; hybrid is the default")] = "hybrid",
    year_from: Annotated[int | None, Field(ge=1870, le=2200)] = None,
    year_to: Annotated[int | None, Field(ge=1870, le=2200)] = None,
    min_vote_average: Annotated[float | None, Field(ge=0.0, le=10.0)] = None,
    genre: Annotated[str | None, Field(max_length=100)] = None,
) -> SemanticSearchResponse:
    """Search movie-vector-app by semantic meaning using Milvus retrieval.

    IMPORTANT: arguments are FLAT. Pass `query` directly, e.g.
    `{"query": "a movie about a blue alien world", "limit": 10}`.
    Do not wrap arguments inside a `request` object.

    The vector app is the sole Milvus owner. Hybrid mode searches grouped metadata,
    overview-only, and title+tagline dense embeddings plus BM25. Use this when the
    request is semantic or as fallback after structured search is weak/no-match.
    Results include retrieval-quality evidence so the agent can judge candidates.
    """
    try:
        request = SemanticSearchRequest(
            query=query,
            limit=limit,
            mode=mode,
            year_from=year_from,
            year_to=year_to,
            min_vote_average=min_vote_average,
            genre=genre,
        )
    except ValueError as exc:
        raise ToolError(f"Invalid semantic search request: {exc}") from exc

    payload = request.model_dump(mode="json", exclude_none=True)
    data = await _request_json(
        "POST",
        f"{settings.vector_app_url.rstrip('/')}/search",
        service_name="movie-vector-app",
        json=payload,
    )
    try:
        return SemanticSearchResponse.model_validate(data)
    except ValueError as exc:
        raise ToolError(f"movie-vector-app returned an invalid search response: {exc}") from exc


@mcp.tool()
async def get_movie_details(
    movie_id: Annotated[int, Field(gt=0, description="TMDB/movie database ID")]
) -> dict:
    """Fetch one complete movie record from movie-db-app.

    Returns movie columns and all normalized related records: genres, keywords,
    production companies/countries, spoken languages, cast credits and crew credits.
    """
    return await _request_json(
        "GET",
        f"{settings.db_app_url.rstrip('/')}/movies/{movie_id}",
        service_name="movie-db-app",
    )


@mcp.tool()
async def resolve_reference_value(
    reference_type: ReferenceType,
    query: Annotated[
        str | None,
        Field(
            description=(
                "User-provided value to normalize. Omit to list canonical values for the reference type."
            ),
            max_length=300,
        ),
    ] = None,
    limit: Annotated[int, Field(ge=1, le=100)] = 20,
) -> ReferenceResolutionResponse:
    """Resolve/validate a canonical value through movie-db-app.

    Supported types include genre, keyword, production company/country,
    spoken language, cast/crew/person, status and original language. This is useful
    before structured search when user wording differs from the canonical DB value.
    """
    payload = {
        "reference_type": reference_type,
        "query": query,
        "limit": min(limit, settings.max_search_limit),
    }
    data = await _request_json(
        "POST",
        f"{settings.db_app_url.rstrip('/')}/agent/references/resolve",
        service_name="movie-db-app",
        json=payload,
    )
    try:
        return ReferenceResolutionResponse.model_validate(data)
    except ValueError as exc:
        raise ToolError(f"movie-db-app returned an invalid reference response: {exc}") from exc


@mcp.custom_route("/health/live", methods=["GET"])
async def health_live(_: Request) -> JSONResponse:
    return JSONResponse({"status": "alive"})


@mcp.custom_route("/health/ready", methods=["GET"])
async def health_ready(_: Request) -> JSONResponse:
    async def healthy(url: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(url)
            return response.status_code == 200
        except Exception:
            return False

    db_ok, vector_ok = await asyncio.gather(
        healthy(f"{settings.db_app_url.rstrip('/')}/health/ready"),
        healthy(f"{settings.vector_app_url.rstrip('/')}/health/ready"),
    )
    ready = db_ok and vector_ok
    return JSONResponse(
        {
            "status": "ready" if ready else "unavailable",
            "db_app": db_ok,
            "vector_app": vector_ok,
        },
        status_code=200 if ready else 503,
    )


if __name__ == "__main__":
    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    mcp.run(
        transport="streamable-http",
        host=settings.mcp_host,
        port=settings.mcp_port,
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        transport_security=security,
    )
