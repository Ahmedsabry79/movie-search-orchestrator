# Movie Agent service boundaries

The stack deliberately has one owner for each datastore and one orchestrator for the agent runtime.

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
                           (vector index source only)

conversation-service --HTTP---------------------------> movie-db-app
                           (conversation/run/event persistence only)
```

## Conversation / agent ownership

`conversation-service` owns the agent loop, but owns no database/vector query implementation.

It provides:

- startup bootstrap of the live MCP tool catalog and authoritative DB schema contract;
- safe high-level planning before execution;
- dynamic prompt construction from the DB logical/search/analytics contract and live MCP instructions;
- startup discovery of every MCP tool, with exact tool JSON schemas supplied natively to Gemma and a compact
  live tool/usage index in the text prompt (avoids duplicating large schemas in the context);
- four planning strategies: `structured_first`, `analytics_first`, `semantic_first`, `direct_answer`;
- required first tool execution for every non-direct plan, preventing unsupported zero-tool catalog answers;
- combined multi-criteria tool calls;
- semantic fallback only after structured movie retrieval when appropriate;
- bounded tool loops (`MAX_AGENT_ITERATIONS`, `MAX_TOOL_CALLS`);
- transient MCP transport retries and a three-consecutive-tool-failure stop policy;
- at most three recent user turns in active model context while full history remains persisted;
- incremental structured long-term memory compaction for turns that age out of the active window;
- retrieval-quality classification for fuzzy DB results and normalized hybrid/dense semantic results;
- one full-cycle context-window recovery using only the immediately previous turn + current request;
- persisted conversations, messages, runs and durable agent events through `movie-db-app`;
- JSON and SSE conversation endpoints with real token-level final-answer streaming;
- safe `thinking`/plan/tool/fallback/recovery events for responsive agent UI;
- no exposure of raw model reasoning or chain-of-thought.

Client endpoints:

- `POST /conversations`
- `GET /conversations/{conversation_id}`
- `POST /conversations/{conversation_id}/messages`
- `POST /conversations/{conversation_id}/messages/stream`

The orchestrator loads `list_tools()` and `get_database_schema` through MCP during service startup.
The resulting context is cached and used by both planner and execution. Exact JSON tool schemas are
provided through the model's native `tools` payload; the system prompt receives the complete logical
DB decision contract plus a compact live catalog of every tool and its MCP usage description. This
keeps capabilities synchronized while avoiding duplicated schema tokens.

Normal active context uses `CONVERSATION_HISTORY_TURNS=3`. Older turns that fall outside that window
are incrementally summarized by an internal no-tools Gemma call into structured long-term memory. The
summary preserves active constraints, corrections, entities/IDs, important results, unresolved questions
and conversational references, then is persisted through `movie-db-app`. The full transcript is never
deleted. Current/recent user messages override compacted memory when they conflict.

On a model context-window rejection, the service emits `context_recovery` and restarts planning/tool
execution from scratch with only the immediately previous completed turn plus the current request
(`RECOVERY_PREVIOUS_TURNS=1`), while still retaining the compacted memory summary.

## Database ownership

`movie-db-app` is the only service that has PostgreSQL credentials or a PostgreSQL driver.

Movie-domain endpoints:

- `POST /agent/search` — comprehensive structured/fuzzy movie retrieval.
- `POST /agent/analyze` — controlled counts/aggregations with allow-listed automatic joins.
- `POST /agent/references/resolve` — canonical/fuzzy reference-value resolution.
- `GET /movies/{movie_id}` — complete movie details and normalized relations.
- `GET /schemas` — schema, structured-search, fuzzy and analytics contract discovery.
- `GET /internal/vector/movies?offset=0&limit=500` — vector index source.

Structured search intentionally excludes `overview`. Plot/overview/theme meaning belongs to semantic
retrieval in `movie-vector-app`.

Fuzzy-capable agent search fields include:

- `title`
- `original_title`
- literal `tagline`
- genre/keyword/company/country/spoken-language relation names
- cast person names
- crew person names
- `cast.character`

Every movie search hit includes `fuzzy_matches` entries containing the fuzzy field, original query,
actual matched DB value and PostgreSQL similarity score. This allows the agent to judge each fuzzy
criterion independently instead of trusting only a single aggregate score.

Controlled analytics supports up to three group dimensions per request. Available dimensions include:

- genre, keyword
- production company, production country
- spoken language
- cast member, crew member, director
- cast character, crew department, crew job
- original language, status, release year

Metrics include distinct movie count plus `sum`, `avg`, `min`, and `max` over budget, revenue,
runtime, vote average, vote count and popularity. The caller never supplies raw SQL, table names or
join expressions; all joins are allow-listed and constructed inside `movie-db-app`.

Conversation persistence endpoints (internal):

- `POST /internal/agent-runtime/conversations`
- `GET /internal/agent-runtime/conversations/{id}`
- `PUT /internal/agent-runtime/conversations/{id}/memory`
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

Hybrid/dense RRF scores are normalized against the theoretical rank-1 maximum for the configured
`RRF_K`, producing a 0..1 `normalized_score`. Results are labeled `strong`, `acceptable`, or `weak`
using configurable semantic thresholds. BM25-only scores remain `unscored` because their scale is
query/corpus dependent. The orchestrator also classifies structured fuzzy results from the minimum
fuzzy similarity across the top candidate's supplied fuzzy criteria. Weak/no-match evidence can open
semantic fallback without being silently presented as a confident match.

## MCP ownership

`mcp-server` is an integration/tool boundary only. It has no SQL driver and no Milvus client.

Tools:

- `get_database_schema` -> `movie-db-app /schemas`
- `search_movies_db` -> `movie-db-app /agent/search`
- `count_movies_db` -> `movie-db-app /agent/analyze`
- `analyze_movies_db` -> `movie-db-app /agent/analyze`
- `resolve_reference_value` -> `movie-db-app /agent/references/resolve`
- `get_movie_details` -> `movie-db-app /movies/{id}`
- `search_movies_semantic` -> `movie-vector-app /search`

`search_movies_semantic` intentionally exposes a flat MCP argument schema (`query`, `limit`, `mode`,
optional filters) instead of a nested request object. The conversation-side MCP client also repairs the
legacy malformed shape `{"request":"..."}` to `{"query":"..."}` before execution.

Tool routing policy:

```text
find/list explicit DB criteria       -> search_movies_db
single distinct-movie count          -> count_movies_db
grouped/aggregate/catalog analytics  -> analyze_movies_db
plot/overview/theme meaning          -> search_movies_semantic
candidate verification               -> get_movie_details
canonical normalization              -> resolve_reference_value
```

## Model gateway

`model-service` provides an OpenAI-compatible gateway:

- `/v1/embeddings` -> BGE-M3 backend
- `/v1/chat/completions` -> Gemma generator backend
- `/v1/completions` -> generator backend
- `/v1/responses` -> generator backend
- `/v1/models` -> merged model listing

The conversation service uses `tool_choice="required"` for the first execution turn of every
non-direct plan, then returns to automatic tool choice after a successful observation. Tool calls and
reasoning fields are passed through to vLLM. Reasoning is preserved internally between tool turns when
returned by vLLM, but never emitted to users.

## Streamlit integration console

`streamlit-app` is a thin manual integration client. It talks only to `conversation-service`; it has
no direct DB, MCP, Milvus, or model access.

```text
Browser
  -> streamlit-app
      -> conversation-service
          -> model-service
          -> mcp-server
              -> movie-db-app -> PostgreSQL
              -> vector-app   -> Milvus
```

It can create/reload persisted conversations, display persisted messages and long-term memory, show
live safe-thinking / plan / tool / retrieval-quality / fallback / memory-compaction / recovery activity,
and render the final assistant answer token-by-token from SSE `assistant_delta` events. Token deltas are
ephemeral; the final assistant message is persisted.

## Deployment cleanup

The Compose stack now uses a fixed project name, loopback-only host port bindings, bounded json-file
log rotation, consistent `unless-stopped` restart behavior, init processes for application containers,
service health checks, and graceful-stop windows. Docker build contexts exclude local caches, virtual
environments, VCS metadata and `.env` files. Existing PostgreSQL/Milvus volumes are preserved on normal
rebuilds; `docker compose down -v` remains intentionally destructive.

## v4.2 tool/UI reliability

- `search_movies_db` is a flat MCP tool schema rather than a single nested `criteria` parameter.
- Conversation orchestration repairs safe legacy/shortcut argument shapes before MCP validation.
- Streamlit SSE is owned by a background worker; UI reruns and sidebar navigation do not terminate generation.
- A conversation is single-flight (one active generation), while separate conversations may generate concurrently.
- The default deployment does not expose PostgreSQL on a host port. `compose.debug-db.yaml` is opt-in for local SQL debugging.

## v4.3 interaction reliability

Structured search sorting is tolerant at both orchestration and MCP boundaries. The preferred form remains `{field, direction}`, while safe shorthand such as `-vote_average`, `+release_year`, and `vote_average:desc` is normalized before `movie-db-app` receives the request. The orchestrator emits the normalized form in `tool_started`, so the visible trace reflects the actual MCP request.

Streamlit keeps SSE generation independent from UI reruns via the existing background worker. Agent activity visibility is now a stable user preference rather than an auto-expanded/auto-collapsed status widget. Once a run is persisted, the UI reconstructs that run's plan/tool/retrieval/recovery trace from `agent_runtime.events` using the assistant message's `metadata.run_id`, so completed execution details remain visible and reloadable with the conversation.
