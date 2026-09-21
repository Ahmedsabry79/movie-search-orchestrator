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
- PostgreSQL host port: `5433`

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
