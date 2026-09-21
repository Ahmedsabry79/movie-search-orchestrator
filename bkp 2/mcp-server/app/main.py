from __future__ import annotations

from typing import Annotated

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
    DatabaseSearchCriteria,
    DatabaseSearchResponse,
    ReferenceResolutionResponse,
    ReferenceType,
    SemanticSearchRequest,
    SemanticSearchResponse,
)


settings = get_settings()

MCP_INSTRUCTIONS = """
Movie-domain retrieval tools for an LLM agent.

Service ownership is strict:
- PostgreSQL is accessed only through movie-db-app.
- Milvus is accessed only through movie-vector-app.
- This MCP server validates tool inputs and calls those service APIs; it contains no SQL
  and no direct Milvus client logic.

Routing policy:
1. If the user supplies a title, original title, recognizable literal tagline text,
   cast name, or crew name, use search_movies_db. Those values are fuzzy matched by
   movie-db-app.
2. Use search_movies_db for structured movie fields and relations: genres, keywords,
   production companies/countries, spoken languages, budget, original language,
   popularity, release date, revenue, runtime, status, vote average/rating, vote count,
   cast/crew, cast character, crew department/job, and combinations of these filters.
3. Use resolve_reference_value when a canonical DB value should be checked or normalized
   first, e.g. genre, spoken language, country, company, person, status, or original language.
4. If the request is semantic rather than a literal database value ("movie about ...",
   "plot where ...", "tagline meant something like ..."), use search_movies_semantic.
5. If structured/fuzzy DB search returns no convincing result, use semantic search as
   fallback. Do not start with semantic search when explicit structured criteria exist.
6. After identifying candidate IDs, use get_movie_details when complete metadata or
   relationship evidence is needed before answering.
7. get_database_schema exposes the authoritative DB schema/search semantics. The conversation
   service preloads this into the system prompt, so do not call it during normal movie search.
""".strip()

mcp = MCPServer("movie-agent-tools", instructions=MCP_INSTRUCTIONS)


def _headers() -> dict[str, str]:
    return {"X-API-Key": settings.api_key} if settings.api_key else {}


def _timeout() -> httpx.Timeout:
    seconds = max(settings.query_timeout_ms / 1000.0, 5.0)
    return httpx.Timeout(seconds, connect=5.0)


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
async def search_movies_db(criteria: DatabaseSearchCriteria) -> DatabaseSearchResponse:
    """Search movie-db-app using one or many structured/fuzzy criteria.

    The DB app owns all PostgreSQL querying. This tool never issues SQL directly.
    Criteria across fields are combined, while relation value sets honor `match=all`
    or `match=any`. Title, original-title, tagline, cast, and crew names are fuzzy
    matched by the DB app. The DB app also searches every normalized relation relevant
    to movies, including genres, keywords, companies, countries, spoken languages,
    cast credits and crew credits.

    Use semantic search instead for broad meaning such as "movie about a stranded
    astronaut" unless a structured DB search first failed and fallback is appropriate.
    """
    criteria.limit = min(criteria.limit, settings.max_search_limit)
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
async def search_movies_semantic(request: SemanticSearchRequest) -> SemanticSearchResponse:
    """Search movie-vector-app by semantic meaning using Milvus hybrid retrieval.

    The vector app is the sole Milvus owner. It searches grouped metadata,
    overview-only, and title+tagline dense embeddings plus BM25 in hybrid mode.
    Use this when the query is semantic or as fallback after structured search fails.
    """
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
