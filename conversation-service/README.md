# Conversation service

Stateful movie-agent orchestrator. It owns planning, context management, tool execution, retrieval-quality policy and streaming, but owns no PostgreSQL or Milvus implementation.

## Flow

1. Persist the user message through `movie-db-app`.
2. Load/cached live MCP tool definitions + DB schema/search contract.
3. Refresh structured long-term memory for turns that have aged out of the active window.
4. Build model context from compacted memory + at most the latest **3 user turns**.
5. Ask Gemma for a concise structured operational plan.
6. Run a bounded native tool-calling loop.
7. Classify retrieval quality and open semantic fallback when structured evidence is weak/no-match.
8. Stream safe execution/status events over SSE.
9. Run a dedicated final-generation phase and stream the answer token-by-token.
10. Persist the final assistant message and complete the run.

## Tool-call repair

`search_movies_semantic` now has a flat MCP schema:

```json
{"query":"a movie about a blue alien world","limit":10,"mode":"hybrid"}
```

The MCP client also repairs the previous malformed legacy shapes:

```json
{"request":"Avatr"}
{"request":{"query":"Avatr","limit":5}}
```

into the flat form before execution. A raw JSON string tool argument is also repaired when the selected tool is unambiguously `search_movies_semantic`.

## Context and long-term memory

Normal active context uses `CONVERSATION_HISTORY_TURNS=3`. The full transcript remains persisted.

When older turns leave that active window, the service incrementally summarizes them with an internal Gemma call that exposes **no MCP tools**. The structured memory preserves:

- active constraints and corrections;
- resolved movie/person entities and IDs;
- established facts and important retrieval outcomes;
- unresolved questions;
- conversational references such as “that movie”.

The summary is persisted through the DB app and injected before recent turns on future requests. Recent user messages always override conflicting memory.

If the model still rejects a request for context length, `context_recovery` restarts the **entire** planning/tool cycle once with only the immediately previous completed turn + current request, while retaining compacted memory.

## Retrieval quality

Structured search still uses DB `FUZZY_THRESHOLD` to collect candidates. The orchestrator then applies stricter evidence thresholds to the top candidate:

```env
FUZZY_ACCEPT_SIMILARITY=0.50
FUZZY_STRONG_SIMILARITY=0.75
```

For hybrid/dense semantic retrieval, the vector service normalizes raw RRF score against the theoretical rank-1 maximum and returns `strong`, `acceptable`, or `weak` quality. Weak results are not treated as confirmed matches.

## Bounded execution

Transient MCP transport errors retry up to `MCP_TRANSPORT_RETRIES` (default 3). Tool-level errors are fed back to Gemma for correction. Three consecutive tool failures stop tool execution and force a grounded graceful response.

## API

```text
POST /conversations
GET  /conversations/{id}
POST /conversations/{id}/messages
POST /conversations/{id}/messages/stream
```

The SSE endpoint can emit:

```text
run_started
memory_compaction_started
memory_compaction_completed
memory_compaction_skipped
thinking
plan
tool_started
tool_result
retrieval_quality
tool_error
fallback_available
context_recovery
response_started
assistant_delta
assistant_message
completed
error
```

`assistant_delta` is ephemeral. The assembled final assistant message is persisted normally. `thinking` is a safe operational status, never raw chain-of-thought.
