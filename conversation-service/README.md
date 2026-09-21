# Conversation service

Stateful movie-agent orchestrator.

## Flow

1. Persist user message through movie-db-app.
2. Load live MCP tool schemas + DB schema context (cached).
3. Ask Gemma for a small structured operational plan.
4. Run a bounded native tool-calling loop.
5. Persist safe plan/tool events through movie-db-app.
6. Persist the final assistant message and complete the run.

Structured-first plans cannot call semantic retrieval before the first DB search. After DB retrieval,
semantic search becomes available for no/weak-match fallback. Semantic-first plans initially cannot
call the structured DB search, preventing accidental SQL-style retrieval for pure plot/theme requests.

Transient MCP transport errors retry up to `MCP_TRANSPORT_RETRIES` (default 3). Tool-level errors are
fed back to the model for argument correction. Three consecutive tool failures stop tool execution and
force a grounded graceful response.

## API

```text
POST /conversations
GET  /conversations/{id}
POST /conversations/{id}/messages
POST /conversations/{id}/messages/stream
```

The streaming endpoint uses SSE events such as:

```text
run_started
plan
tool_started
tool_result
fallback_available
assistant_message
completed
error
```

Raw chain-of-thought/reasoning is never emitted.
