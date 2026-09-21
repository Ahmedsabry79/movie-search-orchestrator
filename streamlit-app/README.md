# Movie Agent Streamlit Test Console

Thin integration UI for the conversation service. It intentionally owns no agent,
database, MCP, Milvus, or model logic.

Flow:

```text
Streamlit -> conversation-service -> model/MCP -> DB App or Vector App
```

The UI can:

- create a persisted conversation;
- load an existing conversation by ID;
- display persisted message history;
- send messages through the SSE endpoint;
- show plan/tool/fallback/error events while the run executes;
- optionally show raw tool payloads and the raw SSE trace;
- display persisted message/run/event counts.

Run with the root compose file and open `http://localhost:8501` by default.
