# Streamlit Integration Console

Thin end-to-end test UI for the movie agent. It talks **only** to `conversation-service`.

## v4.2 streaming model

The SSE request runs in a background worker (`run_manager.py`) rather than on the Streamlit script thread. Streamlit reruns therefore do not terminate an active generation.

While a response is generating you can safely:

- toggle tool payload/debug visibility,
- create and switch to a new conversation,
- load a persisted conversation,
- return to a conversation that is still generating.

The console enforces one active generation per conversation so message ordering remains deterministic. Separate conversations may generate concurrently.

The UI replays safe operational events (plan, tool calls, retrieval quality, fallback, memory compaction, context recovery) and streams the assistant answer incrementally. Raw hidden reasoning is not shown.

## Persistent activity trace

`Show agent activity` is enabled by default and remains stable across Streamlit reruns. Live activity is rendered from the background SSE worker. After completion, activity is rendered from persisted run events associated with the assistant message, so details do not disappear when the final response arrives.
