# Movie Agent Streamlit Integration Console

Thin end-to-end UI. It talks only to `conversation-service` and therefore exercises the real agent path rather than bypassing MCP, DB or vector ownership.

```text
Streamlit -> conversation-service -> model/MCP -> DB App or Vector App
```

The UI can:

- create and reload persisted conversations;
- display persisted chat history;
- inspect the current structured long-term memory;
- show live plan, tool, retrieval-quality, fallback, memory-compaction and recovery activity;
- optionally display tool arguments/results and a bounded SSE trace;
- stream the final answer token-by-token;
- display persisted message/run/event counts.

The activity panel intentionally shows operational summaries rather than raw hidden reasoning.

Run from the root Compose stack and open `http://localhost:8501` by default.
