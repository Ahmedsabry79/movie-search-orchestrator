from __future__ import annotations

from typing import Any

import httpx
import streamlit as st

from client import ClientConfig, ConversationClient
from run_manager import RunManager, RunSnapshot


st.set_page_config(
    page_title="Movie Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { max-width: 1180px; padding-top: 1.55rem; padding-bottom: 3.5rem; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.17); }
      [data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }
      .movie-agent-kicker { font-size: .76rem; opacity: .62; letter-spacing: .09em; text-transform: uppercase; font-weight: 650; }
      .movie-agent-subtle { opacity: .67; font-size: .92rem; max-width: 760px; }
      .movie-agent-hero { margin-bottom: .8rem; }
      .movie-agent-hero h1 { margin: .18rem 0 .15rem 0; font-size: 2.05rem; }
      .movie-agent-pill {
        display: inline-flex; align-items: center; gap: .35rem; padding: .24rem .55rem;
        border-radius: 999px; border: 1px solid rgba(128,128,128,.24); font-size: .78rem;
        margin: 0 .3rem .25rem 0;
      }
      .movie-agent-live-dot {
        width: .45rem; height: .45rem; border-radius: 50%; display: inline-block;
        background: #22c55e; box-shadow: 0 0 0 .16rem rgba(34,197,94,.12);
      }
      .movie-agent-muted { opacity: .62; }
      div[data-testid="stStatusWidget"] { border-radius: 13px; }
      div[data-testid="stChatMessage"] { padding-top: .35rem; padding-bottom: .35rem; }
      div[data-testid="stButton"] button { border-radius: 9px; font-weight: 600; }
      .movie-agent-event-error {
        border-left: 3px solid #ef4444; padding-left: .7rem; margin: .25rem 0 .45rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_client() -> ConversationClient:
    return ConversationClient(ClientConfig.from_env())


def init_state() -> None:
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("conversation_title", None)
    st.session_state.setdefault("load_conversation_id", "")
    st.session_state.setdefault("last_trace", [])
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("show_events", True)
    st.session_state.setdefault("show_payloads", False)
    st.session_state.setdefault("show_debug", False)
    if "run_manager" not in st.session_state:
        st.session_state.run_manager = RunManager()


def manager() -> RunManager:
    return st.session_state.run_manager


def create_new_conversation(client: ConversationClient) -> None:
    record = client.create_conversation(
        title="Movie Agent conversation",
        metadata={"client": "streamlit-integration-console"},
    )
    st.session_state.conversation_id = record["id"]
    st.session_state.conversation_title = record.get("title")
    st.session_state.last_trace = []
    st.session_state.last_error = None


def load_snapshot(client: ConversationClient) -> dict[str, Any] | None:
    conversation_id = st.session_state.conversation_id
    if not conversation_id:
        return None
    try:
        return client.get_conversation(conversation_id)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            st.session_state.conversation_id = None
            st.session_state.conversation_title = None
            st.session_state.last_error = "The selected conversation no longer exists."
            return None
        raise


def _events_by_run(snapshot: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    if not snapshot:
        return grouped
    for event in snapshot.get("events", []):
        run_id = str(event.get("run_id") or "")
        if not run_id:
            continue
        grouped.setdefault(run_id, []).append(event)
    for events in grouped.values():
        events.sort(key=lambda item: (int(item.get("sequence") or 0), str(item.get("created_at") or "")))
    return grouped


def render_persisted_message(
    message: dict[str, Any],
    *,
    events_by_run: dict[str, list[dict[str, Any]]],
    show_events: bool,
    show_payloads: bool,
) -> None:
    role = message.get("role")
    if role not in {"user", "assistant"}:
        return

    with st.chat_message(role):
        if role == "assistant" and show_events:
            metadata = message.get("metadata") or {}
            run_id = str(metadata.get("run_id") or "")
            persisted_events = events_by_run.get(run_id, []) if run_id else []
            if persisted_events:
                with st.container(border=True):
                    st.caption(f"Agent activity · {len(persisted_events)} persisted event(s)")
                    for event in persisted_events:
                        render_trace_event(
                            str(event.get("event_type") or "event"),
                            event.get("payload") or {},
                            show_payloads=show_payloads,
                        )
                st.write("")
        st.markdown(message.get("content") or "")


def compact_tool_result(payload: dict[str, Any]) -> str:
    tool = payload.get("tool", "tool")
    if payload.get("is_error"):
        error = str(payload.get("error") or "Tool execution failed.")
        error = " ".join(error.split())
        if len(error) > 360:
            error = error[:357] + "…"
        return f"{tool} failed — {error}"
    for key in ("returned_count", "count", "matched_movie_count", "total_groups"):
        if payload.get(key) is not None:
            return f"{tool} · {payload[key]} {key.replace('_', ' ')}"
    if payload.get("status"):
        return f"{tool} · {payload['status']}"
    return f"{tool} completed"


def quality_label(payload: dict[str, Any]) -> str:
    source = str(payload.get("source") or "retrieval").replace("_", " ")
    quality = str(payload.get("quality") or "unknown")
    score = payload.get("minimum_top_candidate_similarity")
    if score is None:
        score = payload.get("top_normalized_score")
    suffix = f" · score {float(score):.3f}" if score is not None else ""
    return f"{source.title()} evidence: **{quality}**{suffix}"


def _status_label(run: RunSnapshot) -> tuple[str, str]:
    if run.error:
        return "Agent run failed", "error"
    if run.done:
        return f"Completed · {run.elapsed_seconds:.1f}s", "complete"
    if run.answer:
        return "Writing the answer…", "running"
    if run.events:
        event_type = run.events[-1].get("event")
        payload = run.events[-1].get("data") or {}
        if event_type == "tool_started":
            return f"Calling {payload.get('tool', 'tool')}…", "running"
        if event_type == "thinking":
            return str(payload.get("message") or "Working…"), "running"
        if event_type == "plan":
            return f"Plan · {str(payload.get('strategy') or 'ready').replace('_', ' ')}", "running"
    return "Starting agent run…", "running"


def render_trace_event(event_type: str, payload: dict[str, Any], *, show_payloads: bool) -> None:
    if event_type == "run_started":
        st.caption("Run started")
        return
    if event_type == "thinking":
        st.markdown(f"💭 {payload.get('message') or 'Working on the request…'}")
        return
    if event_type == "memory_compaction_started":
        st.markdown("🧠 **Memory:** compacting older conversation context…")
        return
    if event_type == "memory_compaction_completed":
        st.markdown(
            "🧠 **Memory updated:** "
            f"{payload.get('facts', 0)} facts · {payload.get('constraints', 0)} constraints · "
            f"{payload.get('entities', 0)} entities"
        )
        return
    if event_type == "memory_compaction_skipped":
        st.warning(str(payload.get("message") or "Memory refresh skipped."))
        return
    if event_type == "context_recovery":
        st.warning(str(payload.get("message") or "Restarting with a smaller context…"))
        return
    if event_type == "plan":
        strategy = str(payload.get("strategy") or "unknown").replace("_", " ")
        st.markdown(f"🧭 **Plan · {strategy}:** {payload.get('summary') or 'Plan created'}")
        criteria = payload.get("extracted_criteria") or []
        if criteria:
            st.markdown("**Extracted criteria:** " + " · ".join(str(item) for item in criteria[:8]))
        return
    if event_type == "tool_started":
        st.markdown(f"🔧 **Calling** `{payload.get('tool', 'tool')}`")
        if show_payloads and payload.get("arguments") is not None:
            st.json(payload["arguments"])
        return
    if event_type == "tool_result":
        if payload.get("is_error"):
            st.error(f"Tool result: {compact_tool_result(payload)}")
        else:
            st.markdown(f"✅ **Tool result:** {compact_tool_result(payload)}")
        if show_payloads:
            st.json(payload)
        return
    if event_type == "retrieval_quality":
        quality = str(payload.get("quality") or "unknown")
        icon = {"strong": "🟢", "acceptable": "🟡", "weak": "🟠", "no_match": "🔴"}.get(quality, "⚪")
        st.markdown(f"{icon} {quality_label(payload)}")
        if show_payloads:
            st.json(payload)
        return
    if event_type == "fallback_available":
        st.markdown("↪️ **Fallback:** structured evidence is weak/no-match; semantic retrieval is now available.")
        return
    if event_type == "tool_error":
        st.error(f"Tool failure: `{payload.get('tool', 'unknown')}` — {payload.get('message', 'unknown error')}")
        if show_payloads:
            st.json(payload)
        return
    if event_type == "response_started":
        st.markdown("✍️ **Writing the answer…**")
        return
    if event_type == "completed":
        st.caption(
            f"Completed · {payload.get('tool_calls', 0)} tool call(s) · "
            f"{payload.get('tool_failures', 0)} failure(s)"
        )
        return
    if event_type in {"error", "client_stream_error"}:
        st.error(payload.get("message", "Agent execution failed."))
        return
    if event_type == "assistant_message":
        return
    if show_payloads:
        st.markdown(f"**{event_type}**")
        st.json(payload)


def _latest_user_content(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return None


@st.fragment(run_every=0.35)
def live_run_fragment(
    conversation_id: str,
    persisted_messages: list[dict[str, Any]],
    show_events: bool,
    show_payloads: bool,
) -> None:
    run = manager().get(conversation_id)
    if run is None:
        return

    # The background worker may persist the user message just after the full app
    # rerun. Render it optimistically only when it is not already in the snapshot.
    if _latest_user_content(persisted_messages) != run.prompt:
        with st.chat_message("user"):
            st.markdown(run.prompt)

    with st.chat_message("assistant"):
        label, state = _status_label(run)
        if show_events:
            with st.container(border=True):
                icon = "✅" if state == "complete" else ("❌" if state == "error" else "⏳")
                st.markdown(f"{icon} **Agent activity** · {label}")
                for event in run.events[-80:]:
                    render_trace_event(
                        str(event.get("event") or "event"),
                        event.get("data") or {},
                        show_payloads=show_payloads,
                    )
            st.write("")
        elif not run.done:
            st.caption(label)

        if run.answer:
            st.markdown(run.answer if run.done else run.answer + "▌")
        elif run.error:
            st.error(run.error)
        else:
            st.caption("Waiting for the first response token…")

    if run.done and not run.refresh_requested:
        st.session_state.last_trace = run.events
        manager().mark_refresh_requested(conversation_id)
        # completed is emitted only after the assistant message has been persisted,
        # so a full rerun can safely replace the transient live panel with DB history.
        st.rerun(scope="app")


def sidebar(client: ConversationClient, snapshot: dict[str, Any] | None) -> tuple[bool, bool, bool, bool]:
    run_manager = manager()
    run_manager.prune()

    with st.sidebar:
        st.markdown('<div class="movie-agent-kicker">Integration Console</div>', unsafe_allow_html=True)
        st.title("🎬 Movie Agent")

        ready, health = client.health()
        if ready:
            st.markdown('<span class="movie-agent-pill"><span class="movie-agent-live-dot"></span> Agent ready</span>', unsafe_allow_html=True)
        else:
            st.error("Conversation service unavailable")
            with st.expander("Health details"):
                st.json(health)

        st.write("")
        if st.button("＋ New conversation", use_container_width=True, disabled=not ready, type="primary"):
            try:
                create_new_conversation(client)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not create conversation: {exc}")

        with st.expander("Open conversation", expanded=False):
            st.text_input(
                "Conversation UUID",
                key="load_conversation_id",
                placeholder="Paste a persisted conversation UUID",
                label_visibility="collapsed",
            )
            if st.button("Load", use_container_width=True, disabled=not ready, key="load_conversation_btn"):
                candidate = st.session_state.load_conversation_id.strip()
                if candidate:
                    try:
                        loaded = client.get_conversation(candidate)
                        st.session_state.conversation_id = loaded["conversation"]["id"]
                        st.session_state.conversation_title = loaded["conversation"].get("title")
                        st.session_state.last_trace = []
                        st.session_state.last_error = None
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not load conversation: {exc}")

        active_runs = run_manager.active()
        if active_runs:
            st.divider()
            st.caption("Background generations")
            for active in active_runs:
                selected = active.conversation_id == st.session_state.conversation_id
                st.markdown(
                    f"**{'Current' if selected else 'Running'}** · {active.elapsed_seconds:.1f}s  \n"
                    f"{active.prompt[:72]}{'…' if len(active.prompt) > 72 else ''}"
                )
                if not selected and st.button(
                    "Open live conversation",
                    key=f"open_active_{active.run_id}",
                    use_container_width=True,
                ):
                    st.session_state.conversation_id = active.conversation_id
                    st.session_state.last_error = None
                    st.rerun()

        if st.session_state.conversation_id:
            st.divider()
            st.caption("Current conversation")
            st.code(st.session_state.conversation_id, language=None)
            current_run = run_manager.get(st.session_state.conversation_id)
            if current_run and not current_run.done:
                st.info("Generation continues in the background. You can use sidebar controls without interrupting it.")

        st.divider()
        st.toggle(
            "Show agent activity",
            key="show_events",
            help="Keep planning, tool calls, retrieval quality and recovery events visible during and after generation.",
        )
        st.toggle(
            "Show tool payloads",
            key="show_payloads",
            help="Show full tool arguments/results inside the activity trace.",
        )
        st.toggle(
            "Show debug trace",
            key="show_debug",
            help="Show the raw recent SSE event list in the sidebar for debugging.",
        )

        if snapshot:
            messages = snapshot.get("messages", [])
            runs = snapshot.get("runs", [])
            events = snapshot.get("events", [])
            c1, c2, c3 = st.columns(3)
            c1.metric("Msgs", len(messages))
            c2.metric("Runs", len(runs))
            c3.metric("Events", len(events))

            conversation = snapshot.get("conversation") or {}
            memory = conversation.get("memory_summary")
            if memory:
                with st.expander("Long-term memory", expanded=False):
                    st.json(memory)

            if runs:
                latest = runs[0]
                st.caption(f"Latest persisted run · {latest.get('status', 'unknown')}")

        if st.session_state.show_debug and st.session_state.last_trace:
            with st.expander("Last SSE trace", expanded=False):
                st.json(st.session_state.last_trace)

    return (
        ready,
        st.session_state.show_events,
        st.session_state.show_payloads,
        st.session_state.show_debug,
    )


def main() -> None:
    init_state()
    client = get_client()
    manager().prune()

    ready, _ = client.health()
    if ready and not st.session_state.conversation_id:
        try:
            create_new_conversation(client)
        except Exception as exc:
            st.session_state.last_error = str(exc)

    try:
        snapshot = load_snapshot(client) if st.session_state.conversation_id else None
    except Exception as exc:
        snapshot = None
        st.session_state.last_error = str(exc)

    # If a completed background run is now present in persisted history, retire its
    # transient UI object. The conversation remains fully available from PostgreSQL.
    current_id = st.session_state.conversation_id
    current_run = manager().get(current_id)
    if current_run and current_run.done and snapshot:
        assistant_messages = [
            str(item.get("content") or "")
            for item in snapshot.get("messages", [])
            if item.get("role") == "assistant"
        ]
        if current_run.answer and current_run.answer in assistant_messages:
            manager().clear_completed(current_id)
            current_run = None

    ready, show_events, show_payloads, _ = sidebar(client, snapshot)

    st.markdown(
        """
        <div class="movie-agent-hero">
          <div class="movie-agent-kicker">Agentic Movie Retrieval</div>
          <h1>Movie Agent</h1>
          <div class="movie-agent-subtle">
            Structured and fuzzy catalog search, controlled analytics, semantic retrieval,
            persisted memory, and live tool execution.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    persisted_messages = snapshot.get("messages", []) if snapshot else []
    persisted_events_by_run = _events_by_run(snapshot)
    for message in persisted_messages:
        render_persisted_message(
            message,
            events_by_run=persisted_events_by_run,
            show_events=show_events,
            show_payloads=show_payloads,
        )

    current_run = manager().get(st.session_state.conversation_id)
    if current_run:
        live_run_fragment(
            st.session_state.conversation_id,
            persisted_messages,
            show_events,
            show_payloads,
        )

    if not snapshot and ready:
        st.info("Create or load a conversation to begin.")

    busy = bool(current_run and not current_run.done)
    prompt = st.chat_input(
        "Ask about a movie, describe a plot, combine filters, or query the catalog…",
        disabled=(
            not ready
            or not bool(st.session_state.conversation_id)
            or busy
        ),
    )

    if busy:
        st.caption("This conversation is generating. Sidebar controls remain available; open a new conversation to ask another question in parallel.")

    if prompt:
        try:
            manager().start(client, st.session_state.conversation_id, prompt)
            st.session_state.last_error = None
            st.rerun()
        except Exception as exc:
            st.session_state.last_error = str(exc)
            st.rerun()

    if snapshot and not persisted_messages and not current_run:
        st.caption(
            "Try: “How many movies are there in each genre?” · “Find the movie about entering dreams” · "
            "“Nolan movies after 2010 rated above 8”"
        )


if __name__ == "__main__":
    main()
