# Agentic Movie Retrieval Platform

End-to-end movie discovery and analysis platform built around deterministic service ownership, bounded orchestration, and grounded retrieval.

## Design principles

The system is intentionally designed around explicit application logic rather than unrestricted model behavior.

The LLM is used for:

- intent interpretation,
- planning,
- tool selection,
- conversational continuity,
- and grounded response generation.

Deterministic application components remain responsible for:

- database access,
- structured filtering,
- fuzzy matching,
- analytics,
- vector retrieval,
- BM25 retrieval,
- rank fusion,
- validation,
- retries,
- persistence,
- memory lifecycle,
- service ownership,
- and API boundaries.

The model has no direct access to PostgreSQL or Milvus and is not treated as a source of movie facts.

## Architecture

Centralized data ownership is enforced across the stack:

- `movie-db-app`: sole PostgreSQL owner; seeding, structured/fuzzy search, analytics, movie details, and agent-runtime persistence.
- `movie-vector-app`: sole Milvus owner; indexing and hybrid/dense/BM25 retrieval.
- `model-service`: OpenAI-compatible gateway for Gemma generation and BGE-M3 embeddings.
- `mcp-server`: typed tool boundary; calls DB/vector services and contains no SQL or Milvus client.
- `conversation-service`: planning, bounded tool execution, retrieval-quality policy, streaming, persistence coordination, and memory management.
- `streamlit-app`: thin end-to-end integration UI; communicates only with `conversation-service`.

## Start

```bash
cp .env.example .env

# Set HUGGING_FACE_HUB_TOKEN and any non-default credentials/options.
docker compose up -d --build
```

Open Streamlit at:

```text
http://localhost:8501
```

Useful local endpoints:

- DB app: `8001`
- model gateway: `8002`
- vector app: `8003`
- MCP server: `8004`
- conversation service: `8005`
- Streamlit: `8501`

## Upgrade / rebuild

For code-only changes, keep persistent volumes:

```bash
docker compose up -d --build --force-recreate \
  app vector-app mcp-server conversation-service streamlit-app
```

Do **not** use:

```bash
docker compose down -v
```

unless PostgreSQL and Milvus data should intentionally be deleted.

The DB app applies idempotent additive migrations for conversation-memory fields during startup.

## Health

```bash
curl -f http://localhost:8001/health/ready
curl -f http://localhost:8002/health/ready
curl -f http://localhost:8003/health/ready
curl -f http://localhost:8004/health/ready
curl -f http://localhost:8005/health/ready
curl -f http://localhost:8501/_stcore/health
```

## Retrieval architecture

The retrieval stack separates structured, fuzzy, semantic, and lexical responsibilities.

Structured catalog constraints are handled by PostgreSQL through the DB app.

Semantic retrieval uses three independent dense movie representations:

- `dense_grouped`
  - title
  - original title
  - tagline
  - genres
  - keywords
  - overview
- `dense_overview`
  - overview only
- `dense_title_tagline`
  - title + tagline

BM25 runs over the grouped movie document.

Hybrid retrieval combines:

```text
dense_grouped
dense_overview
dense_title_tagline
BM25
        ↓
       RRF
        ↓
 final ranked results
```

Raw BM25 and embedding scores are not directly added together. Reciprocal Rank Fusion combines channel rankings, avoiding incompatible raw-score scales.

There is no dedicated cross-encoder reranker in the current implementation. RRF is used as the lightweight deterministic fusion stage.

## Retrieval thresholds

Structured search uses the database candidate-inclusion threshold:

```text
FUZZY_THRESHOLD=0.30
```

The orchestration layer then evaluates the strongest returned fuzzy evidence using stricter quality thresholds:

```text
FUZZY_ACCEPT_SIMILARITY=0.50
FUZZY_STRONG_SIMILARITY=0.75
```

Hybrid and dense RRF scores are normalized against the theoretical rank-1 maximum for the configured RRF setup before quality classification:

```text
SEMANTIC_ACCEPT_NORMALIZED_SCORE=0.25
SEMANTIC_STRONG_NORMALIZED_SCORE=0.45
```

These values are matching-quality heuristics, not probabilities. They are configurable and should be calibrated against a larger labeled evaluation set for production deployment.

## Semantic metadata filters

The vector retrieval path currently supports deterministic metadata filtering for:

- genre,
- release-year bounds,
- minimum vote average.

Other strict constraints remain owned by the structured DB path.

For example:

- cast,
- crew,
- production company,
- runtime,
- budget,
- revenue,
- language,
- character,
- and other relational/scalar conditions

are verified through PostgreSQL rather than assumed from semantic similarity.

Keywords are not exposed as strict Milvus metadata filters, but they participate in retrieval through:

- the grouped dense representation,
- and the BM25 document.

## Conversation memory

The complete conversation transcript is persisted in PostgreSQL.

Active generation uses at most three recent user turns by default, including the current request and the available assistant replies for those turns.

Older turns are incrementally compacted into structured long-term memory.

The memory representation preserves items such as:

- active constraints,
- corrections,
- resolved entities,
- important results,
- unresolved questions,
- and conversational references.

Memory compaction is performed as an isolated summarization step with tool access disabled.

Relevant settings:

```text
CONVERSATION_HISTORY_TURNS=3
MEMORY_COMPACTION_ENABLED=true
MEMORY_COMPACTION_MIN_MESSAGES=2
MEMORY_COMPACTION_MAX_TOKENS=1000
MEMORY_COMPACTION_MESSAGE_LIMIT=200
MEMORY_COMPACTION_MAX_CHARS=16000
```

Recent conversation messages always take precedence over compacted memory.

The full transcript remains persisted even when older turns are no longer carried verbatim in active model context.

## Agent execution model

Each user request runs through a bounded orchestration loop.

The conversation service is responsible for:

- planning,
- strategy selection,
- tool availability,
- argument normalization,
- tool retries,
- retrieval-quality evaluation,
- bounded replanning,
- persistence,
- and final response streaming.

The model may select tools, but execution remains constrained by typed MCP contracts and application-side limits.

Default execution limits:

```text
MAX_TOOL_FAILURES=3
MAX_TOOL_CALLS=10
MAX_AGENT_ITERATIONS=8
```

Repeated failures are bounded rather than retried indefinitely.

When the context window is rejected for length, the conversation service contains a complete-cycle recovery path that retries with reduced recent context while retaining compact memory.

## MCP tool contract

The MCP server is the model-facing integration boundary.

It exposes typed movie-domain tools while remaining independent of storage implementation.

Primary tools include:

```text
get_database_schema
search_movies_db
count_movies_db
analyze_movies_db
resolve_reference_value
get_movie_details
search_movies_semantic
```

The MCP server contains:

- no PostgreSQL connection,
- no SQL,
- no Milvus client.

It delegates to the owning services over HTTP.

## Semantic tool contract

`search_movies_semantic` uses a flat MCP schema.

Example:

```json
{
  "query": "a movie about a blue alien world",
  "limit": 10,
  "mode": "hybrid"
}
```

Supported modes:

```text
hybrid
dense
bm25
```

The conversation service also includes backward-compatibility normalization for the previous request shape:

```json
{
  "request": "..."
}
```

## Structured-search MCP contract

`search_movies_db` also uses a flat MCP schema.

Do not wrap arguments inside a `criteria` object.

Example:

```json
{
  "titles": ["Avatr"],
  "cast": {
    "values": ["Sam Worthington"],
    "match": "all"
  },
  "release_year": {
    "min": 2000
  },
  "vote_average": {
    "min": 7.0
  }
}
```

The conversation service includes compatibility normalization for legacy and supported shorthand forms before MCP validation.

Supported normalization includes:

```text
{"criteria": {...}}
    → flattened structured arguments

{"criteria": "Avatr"}
    → {"titles": ["Avatr"]}

relation string/list
    → {"values": [...], "match": "all"}

gte/lte
    → min/max

scalar numeric range
    → eq

lone generic query in search_movies_db
    → title fuzzy search
```

Failed tool cards in Streamlit display the bounded MCP/HTTP validation error rather than hiding the underlying failure.

## Search sort normalization

`search_movies_db.sort` accepts canonical structured sort objects and supported shorthand forms.

These are equivalent:

```json
{
  "sort": [
    {
      "field": "vote_average",
      "direction": "desc"
    }
  ]
}
```

```json
{
  "sort": "-vote_average"
}
```

```json
{
  "sort": ["-vote_average"]
}
```

```json
{
  "sort": "vote_average:desc"
}
```

The conversation service normalizes supported sort forms before tracing and execution.

The MCP tool also accepts the shorthand directly.

This prevents avoidable validation failures when a non-canonical sort representation is received.

## Streamlit execution model

The Streamlit application is intentionally thin.

It communicates only with the conversation service and does not directly access:

- PostgreSQL,
- Milvus,
- MCP,
- or the model service.

This ensures that manual testing exercises the same complete application path as any future API client.

### Non-blocking streaming

The Streamlit UI does not keep the SSE request on the main UI script thread.

A background worker maintains the conversation-service stream while Streamlit reruns independently.

This allows the following controls to remain usable while generation is active:

- New conversation
- Load conversation
- Show/hide tool payloads
- Show/hide debug trace
- Switch back to a conversation that is still generating

Only one active generation is allowed per conversation to preserve message ordering.

Different conversations may generate concurrently.

### Persistent activity trace

Agent activity is controlled by the persistent `Show agent activity` preference, enabled by default.

Completed assistant messages reconstruct their activity trace from persisted `agent_runtime.events`.

This keeps operational execution details available after:

- generation completes,
- normal Streamlit reruns,
- switching conversations,
- returning to a previous conversation,
- or restarting Streamlit.

Persisted activity includes safe operational events such as:

- planning,
- tool calls,
- tool results,
- retrieval quality,
- fallback,
- recovery,
- and execution status.

Raw hidden model reasoning is not exposed.

Tool-payload and debug toggles use stable `st.session_state` keys and can be changed while an SSE worker is active without terminating generation.

## PostgreSQL host exposure

The default Compose stack does not publish PostgreSQL to the host.

Application services communicate internally using:

```text
db:5432
```

For optional host-side SQL debugging:

```bash
docker compose \
  -f compose.yaml \
  -f compose.debug-db.yaml \
  up -d db
```

The debug overlay exposes:

```text
127.0.0.1:${POSTGRES_HOST_PORT:-5434}:5432
```

## API security

Service APIs support optional shared API-key protection for local/internal deployment.

The assessment configuration is intentionally lightweight and does not implement a complete production identity layer.

A production deployment should additionally introduce:

- mandatory authentication,
- role-based authorization,
- secret rotation,
- TLS termination,
- rate limiting,
- per-user access control,
- and dedicated secrets management.

## Data semantics

Missing numeric values are not automatically interpreted as meaningful zeros.

For example:

- `budget <= 0` → unknown
- `revenue <= 0` → unknown
- `vote_count <= 0` → unknown
- `runtime <= 0` → unknown
- `vote_average <= 0` → unknown

Popularity is handled differently because zero can be a valid value.

This prevents missing monetary data from being treated as genuine zero-cost or zero-revenue values during filtering and analytics.

## Reliability notes

The system intentionally separates transport failures from domain and validation failures.

Transport failures may be retried with bounded backoff.

Validation/domain failures are returned as observations so arguments can be repaired without silently changing query semantics.

Three consecutive tool failures stop further tool execution for the run.

The system then generates a grounded response using only evidence already retrieved successfully.

## Version notes

### v4.1 — memory persistence hotfix

Fixed conversation-message persistence after the memory-compaction update.

Memory fields belong to:

```text
agent_runtime.conversations
```

Message serialization no longer attempts to read them from:

```text
agent_runtime.messages
```

### v4.2 — reliability and UI update

Changes include:

- flat structured-search MCP contract,
- structured argument normalization,
- clearer validation failures,
- non-blocking Streamlit streaming,
- concurrent background generation across conversations,
- PostgreSQL no longer exposed to the host by default.

### v4.3 — sort normalization and persistent activity trace

Changes include:

- canonical and shorthand sort normalization,
- persistent agent-activity visibility,
- reconstruction of completed execution traces from persisted events,
- stable tool/debug UI state during active streaming.

## Production considerations

The current implementation prioritizes correctness, clear ownership boundaries, reproducibility, and grounded retrieval over unnecessary feature breadth.

Known areas for future improvement include:

- broader vector-side metadata filtering,
- learned cross-encoder reranking,
- larger retrieval-threshold evaluation sets,
- production-grade authentication and authorization,
- public deployment infrastructure,
- live movie-data ingestion,
- and optional external-data fallback where appropriate.
