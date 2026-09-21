# Movie Agent Stack

End-to-end movie catalog agent with centralized data ownership:

- `movie-db-app`: sole PostgreSQL owner; seeding, structured/fuzzy search, analytics, movie details, agent persistence.
- `movie-vector-app`: sole Milvus owner; indexing and hybrid/dense/BM25 retrieval.
- `model-service`: OpenAI-compatible gateway for Gemma generation and BGE-M3 embeddings.
- `mcp-server`: tool boundary; calls DB/vector apps and contains no SQL or Milvus client.
- `conversation-service`: planning, bounded tool loop, streaming, memory compaction, retrieval-quality policy.
- `streamlit-app`: thin end-to-end test UI; talks only to `conversation-service`.

## Start

```bash
cp .env.example .env
# Set HUGGING_FACE_HUB_TOKEN and any non-default credentials/options.
docker compose up -d --build
```

Open Streamlit at `http://localhost:8501` by default.

Useful local endpoints (all bind to loopback by default):

- DB app: `8001`
- model gateway: `8002`
- vector app: `8003`
- MCP server: `8004`
- conversation service: `8005`
- Streamlit: `8501`

## Upgrade / rebuild

For code-only changes, keep volumes:

```bash
docker compose up -d --build --force-recreate \
  app vector-app mcp-server conversation-service streamlit-app
```

Do **not** use `docker compose down -v` unless you intentionally want to delete PostgreSQL and Milvus data.

The DB app performs an idempotent additive migration for conversation-memory columns on startup.

## Health

```bash
curl -f http://localhost:8001/health/ready
curl -f http://localhost:8002/health/ready
curl -f http://localhost:8003/health/ready
curl -f http://localhost:8004/health/ready
curl -f http://localhost:8005/health/ready
curl -f http://localhost:8501/_stcore/health
```

## Retrieval thresholds

Structured search still uses the DB inclusion threshold (`FUZZY_THRESHOLD`, default `0.30`) to gather candidates. The agent then classifies the top candidate using stricter evidence thresholds:

- `FUZZY_ACCEPT_SIMILARITY=0.50`
- `FUZZY_STRONG_SIMILARITY=0.75`

Hybrid/dense Milvus RRF scores are normalized to a 0–1 theoretical rank-1 maximum before classification:

- `SEMANTIC_ACCEPT_NORMALIZED_SCORE=0.25`
- `SEMANTIC_STRONG_NORMALIZED_SCORE=0.45`

These are orchestration-quality thresholds, not probabilities. Tune them against the evaluation/query set if needed.

## Conversation memory

The complete transcript is persisted in PostgreSQL. Active generation uses at most three recent user turns by default. Older expired turns are incrementally summarized into structured long-term memory and persisted through the DB app. Memory compaction is an internal model call with no MCP tools.

Relevant settings:

```env
CONVERSATION_HISTORY_TURNS=3
MEMORY_COMPACTION_ENABLED=true
MEMORY_COMPACTION_MIN_MESSAGES=2
MEMORY_COMPACTION_MAX_TOKENS=1000
MEMORY_COMPACTION_MESSAGE_LIMIT=200
MEMORY_COMPACTION_MAX_CHARS=16000
```

## Semantic tool contract

`search_movies_semantic` has a flat MCP schema. Example:

```json
{
  "query": "a movie about a blue alien world",
  "limit": 10,
  "mode": "hybrid"
}
```

The conversation service also contains a compatibility repair for the previous malformed shape `{"request":"..."}`.

## v4.1 hotfix

Fixes conversation message persistence after the memory-compaction update. Memory fields belong to `agent_runtime.conversations`; message serialization no longer attempts to read them from `agent_runtime.messages`.

## v4.2 reliability / UI update

### Structured-search MCP contract

`search_movies_db` now has a flat MCP schema. Do not wrap arguments in a `criteria` object.

```json
{
  "titles": ["Avatr"],
  "cast": {"values": ["Sam Worthington"], "match": "all"},
  "release_year": {"min": 2000},
  "vote_average": {"min": 7.0}
}
```

The conversation service keeps compatibility with older/generated shapes and safely repairs common unambiguous shortcuts before MCP validation, including:

- `{"criteria": {...}}` -> flattened search arguments
- `{"criteria": "Avatr"}` -> `{"titles": ["Avatr"]}`
- relation string/list -> `{values: [...], match: "all"}`
- `gte/lte` -> `min/max`
- scalar numeric range -> `eq`
- a lone generic `query` on `search_movies_db` -> title fuzzy search

Failed tool cards in Streamlit now show the actual bounded MCP/HTTP validation error by default.

### Non-blocking Streamlit streaming

The Streamlit integration console no longer owns the SSE connection on the UI script thread. A background worker keeps the conversation-service stream alive while Streamlit reruns independently.

This means the following controls can be used while another conversation is generating without cancelling or duplicating the active request:

- New conversation
- Load conversation
- Show/hide tool payloads
- Show/hide debug trace
- Switch back to a conversation that is still generating

Only one generation is allowed at a time per conversation, preserving message ordering. Different conversations can have background generations concurrently.

### PostgreSQL host exposure

The default Compose stack no longer publishes PostgreSQL to the host. All app services use `db:5432` internally.

For optional host-side SQL debugging:

```bash
docker compose -f compose.yaml -f compose.debug-db.yaml up -d db
```

The overlay uses `127.0.0.1:${POSTGRES_HOST_PORT:-5434}:5432`.

## v4.3 sort + persistent activity trace

### Search sort normalization

`search_movies_db.sort` now accepts either canonical structured sort objects or compact shorthand. These are equivalent:

```json
{"sort": [{"field": "vote_average", "direction": "desc"}]}
```

```json
{"sort": "-vote_average"}
```

```json
{"sort": ["-vote_average"]}
```

```json
{"sort": "vote_average:desc"}
```

The conversation service normalizes these shapes before tracing/execution, and the MCP tool itself also accepts the shorthand directly. This prevents Pydantic validation failures when the model emits a string sort.

### Streamlit activity persistence

Agent activity is now controlled by a persistent `Show agent activity` preference (enabled by default). The UI no longer auto-collapses the live event panel when generation completes.

Completed assistant messages reconstruct their activity trace from persisted `agent_runtime.events`, so planning/tool/retrieval/recovery details remain reviewable after:

- the final response is ready,
- a normal Streamlit rerun,
- loading another conversation and returning,
- a Streamlit restart.

Tool payload and debug toggles use their own stable `st.session_state` keys and can be changed while an SSE worker is active without terminating generation.
