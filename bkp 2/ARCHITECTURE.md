# Movie Agent service boundaries

The stack deliberately has one owner for each data store and one orchestrator for the agent runtime.

```text
Client / UI
    |
    v
conversation-service
    |  \
    |   +-----------------------> model-service ------> Gemma 4
    |
    +---------------------------> MCP server
                                      |      |
                                      |      +----------> movie-vector-app -----> Milvus
                                      |
                                      +-----------------> movie-db-app ---------> PostgreSQL

movie-vector-app --HTTP-------------------------------> movie-db-app
                           (index source only)

conversation-service --HTTP---------------------------> movie-db-app
                           (conversation/run/event persistence only)
```

## Conversation / agent ownership

`conversation-service` owns the agent loop, but owns no database/vector query implementation.

It provides:

- safe high-level planning before execution;
- dynamic prompt construction from the DB schema and live MCP tool schemas;
- structured-first vs semantic-first routing;
- combined multi-criteria tool calls;
- semantic fallback after structured retrieval has been observed;
- bounded tool loops (`MAX_AGENT_ITERATIONS`, `MAX_TOOL_CALLS`);
- transient MCP transport retries and a three-consecutive-tool-failure stop policy;
- persisted conversations, messages, runs and tool events through `movie-db-app`;
- JSON and SSE conversation endpoints;
- no exposure of raw model reasoning; UI traces contain safe plans/tool events only.

Client endpoints:

- `POST /conversations`
- `GET /conversations/{conversation_id}`
- `POST /conversations/{conversation_id}/messages`
- `POST /conversations/{conversation_id}/messages/stream`

The orchestrator fetches `get_database_schema` through MCP and compacts all movie-domain tables,
columns, relationships, search fields and semantics into the system context. It also injects the
MCP server instructions and every live MCP tool description/input JSON schema. The prompt therefore
stays synchronized with DB/MCP contract changes without hardcoding DDL in the conversation service.

## Database ownership

`movie-db-app` is the only service that has PostgreSQL credentials or a PostgreSQL driver.

Movie-domain endpoints:

- `POST /agent/search` — comprehensive structured/fuzzy movie search.
- `POST /agent/references/resolve` — canonical/fuzzy reference-value resolution.
- `GET /movies/{movie_id}` — complete movie details and all normalized relations.
- `GET /schemas` — schema/search-contract discovery.
- `GET /internal/vector/movies?offset=0&limit=500` — vector index source.

Conversation persistence endpoints (internal):

- `POST /internal/agent-runtime/conversations`
- `GET /internal/agent-runtime/conversations/{id}`
- `POST /internal/agent-runtime/conversations/{id}/messages`
- `POST /internal/agent-runtime/conversations/{id}/runs`
- `PATCH /internal/agent-runtime/runs/{id}`
- `POST /internal/agent-runtime/runs/{id}/events`

Runtime data lives in a separate `agent_runtime` PostgreSQL schema so it is not part of the immutable
TMDB seed/version accounting in `movie_catalog`.

## Vector ownership

`movie-vector-app` is the only service importing `pymilvus` / creating a `MilvusClient`.

It owns:

- Milvus collection creation and schema;
- index/reindex lifecycle;
- BGE-M3 embedding requests for indexing/query retrieval;
- three dense fields per movie (`dense_grouped`, `dense_overview`, `dense_title_tagline`);
- Milvus BM25 sparse field;
- dense/BM25/RRF hybrid search.

It has no PostgreSQL credentials. Index source data is fetched over HTTP from `movie-db-app`.

## MCP ownership

`mcp-server` is an integration/tool boundary only. It has no SQL driver and no Milvus client.

Tools:

- `get_database_schema` -> `movie-db-app /schemas`
- `search_movies_db` -> `movie-db-app /agent/search`
- `resolve_reference_value` -> `movie-db-app /agent/references/resolve`
- `get_movie_details` -> `movie-db-app /movies/{id}`
- `search_movies_semantic` -> `movie-vector-app /search`

Structured/fuzzy search is preferred when explicit database criteria are present. Semantic search is
used for meaning/plot/tagline-description queries and becomes available as fallback after structured
search for structured-first runs.

## Model gateway

`model-service` provides an OpenAI-compatible gateway:

- `/v1/embeddings` -> BGE-M3 backend
- `/v1/chat/completions` -> Gemma generator backend
- `/v1/completions` -> generator backend
- `/v1/responses` -> generator backend
- `/v1/models` -> merged model listing

Tool calls/reasoning fields are passed through unchanged to vLLM. The conversation service preserves
reasoning internally between tool turns when returned by vLLM, but never emits raw reasoning to users.
