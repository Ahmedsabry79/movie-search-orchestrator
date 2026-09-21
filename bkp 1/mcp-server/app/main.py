from __future__ import annotations

import asyncio
from typing import Annotated

import httpx
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import get_settings
from .db import MovieDatabaseRepository
from .schemas import (
    DatabaseSearchCriteria,
    DatabaseSearchResponse,
    ReferenceResolutionResponse,
    ReferenceType,
    SemanticSearchRequest,
    SemanticSearchResponse,
)


settings = get_settings()
repository = MovieDatabaseRepository(
    settings.postgres_dsn,
    fuzzy_threshold=settings.fuzzy_threshold,
    query_timeout_ms=settings.query_timeout_ms,
)


MCP_INSTRUCTIONS = """
Movie-domain retrieval tools for an LLM agent.

Routing policy:
1. If the user supplies a title, original title, recognizable tagline text, cast name,
   or crew name, use search_movies_db. Those fields are fuzzy matched.
2. Use search_movies_db for structured movie fields and relations: genres, keywords,
   production companies/countries, spoken languages, budget, original language,
   popularity, release date, revenue, runtime, status, rating/vote average, vote count,
   cast/crew, character, crew department/job, and combinations of these filters.
3. Use resolve_reference_value when a canonical database value should be checked or
   normalized first (for example a genre, spoken language, country, company, or person).
4. If the request is semantic rather than a literal database value ("movie about ...",
   "plot where ...", "tagline meant something like ..."), use search_movies_semantic.
5. If a structured/fuzzy search returns no convincing result, semantic search is the
   fallback. Do not use semantic search first when explicit structured criteria exist.
6. After identifying candidate IDs, use get_movie_details when complete metadata or
   relationship evidence is needed before answering.
""".strip()

mcp = MCPServer("movie-agent-tools", instructions=MCP_INSTRUCTIONS)


@mcp.tool()
async def search_movies_db(criteria: DatabaseSearchCriteria) -> DatabaseSearchResponse:
    """Search PostgreSQL using one or many structured movie criteria.

    This is the primary tool for literal/structured constraints. It supports every
    normalized movie relation used by the dataset: genres, keywords, production
    companies, production countries, spoken languages, cast and crew. It also supports
    movie-table filters including budget, original language, popularity, release date,
    revenue, runtime, status, vote average/rating and vote count.

    `titles`, `original_titles`, `taglines`, `cast`, and `crew` use fuzzy matching so
    misspellings and close names can still match. Relation filters combine with movie
    column filters using AND. Within a relation, `match=all` requires every supplied
    value and `match=any` accepts any supplied value.

    Do not put broad semantic descriptions such as "movie about a stranded astronaut"
    into title/tagline fields. Use search_movies_semantic for semantic meaning.
    """
    try:
        criteria.limit = min(criteria.limit, settings.max_search_limit)
        rows = await asyncio.to_thread(repository.search, criteria)
        return DatabaseSearchResponse(
            count=len(rows),
            criteria=criteria.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
            semantic_fallback_recommended=(len(rows) == 0),
            results=rows,
        )
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Database movie search failed: {exc}") from exc


@mcp.tool()
async def search_movies_semantic(request: SemanticSearchRequest) -> SemanticSearchResponse:
    """Search the movie vector index by semantic meaning, with dense + BM25 retrieval.

    Use this when the user describes plot/theme/meaning rather than supplying a literal
    database value, e.g. "a movie about memory manipulation" or "the tagline meant that
    humanity had to leave Earth". It is also the fallback when a structured/fuzzy DB
    search found no good match.

    The vector index compares the same query embedding independently against grouped
    movie metadata, overview-only, and title+tagline embeddings, then fuses those with
    BM25 in hybrid mode.
    """
    payload = request.model_dump(exclude_none=True)
    headers = {"X-API-Key": settings.api_key} if settings.api_key else {}
    try:
        timeout = max(settings.query_timeout_ms / 1000.0, 5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{settings.vector_app_url.rstrip('/')}/search",
                json=payload,
                headers=headers,
            )
        if response.status_code >= 400:
            raise ToolError(
                f"Semantic movie search failed with HTTP {response.status_code}: {response.text[:500]}"
            )
        data = response.json()
        return SemanticSearchResponse.model_validate(data)
    except ToolError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise ToolError(f"Semantic movie search failed: {exc}") from exc


@mcp.tool()
async def get_movie_details(
    movie_id: Annotated[int, Field(gt=0, description="TMDB/movie database ID")]
) -> dict:
    """Fetch one movie with all normalized related records.

    Returns the movie row plus genres, keywords, production companies, production
    countries, spoken languages, full cast credits (including character/order), and
    full crew credits (including department/job). Use this to verify candidates or to
    answer detailed follow-up questions after search.
    """
    try:
        result = await asyncio.to_thread(repository.movie_details, movie_id)
        if result is None:
            raise ToolError(f"Movie ID {movie_id} was not found")
        return result
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Movie details lookup failed: {exc}") from exc


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
    """Validate or normalize canonical movie database values before structured search.

    Supported reference types: genre, keyword, production_company,
    production_country, spoken_language, cast, crew, person, status and
    original_language. The resolver uses the normalized DB values and fuzzy matching
    where appropriate. For example, resolve `sci fi` to `Science Fiction` before
    applying a genre filter, or validate a spoken language/country/company/person.

    If `query` is omitted, returns canonical values (bounded by `limit`).
    """
    try:
        rows = await asyncio.to_thread(
            repository.resolve_reference,
            reference_type,
            query,
            limit=min(limit, settings.max_search_limit),
        )
        return ReferenceResolutionResponse(
            reference_type=reference_type,
            query=query,
            count=len(rows),
            results=rows,
        )
    except Exception as exc:
        raise ToolError(f"Reference resolution failed: {exc}") from exc


@mcp.custom_route("/health/live", methods=["GET"])
async def health_live(_: Request) -> JSONResponse:
    return JSONResponse({"status": "alive"})


@mcp.custom_route("/health/ready", methods=["GET"])
async def health_ready(_: Request) -> JSONResponse:
    db_ok = await asyncio.to_thread(repository.ping)
    vector_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{settings.vector_app_url.rstrip('/')}/health/ready")
            vector_ok = response.status_code == 200
    except Exception:
        vector_ok = False
    ready = db_ok and vector_ok
    return JSONResponse(
        {"status": "ready" if ready else "unavailable", "postgres": db_ok, "vector_app": vector_ok},
        status_code=200 if ready else 503,
    )


if __name__ == "__main__":
    # This service is only exposed on the internal Docker network and, optionally,
    # host loopback. Disable DNS-rebinding filtering so Docker service names such as
    # `mcp-server:8000` work without an external-host allowlist.
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
