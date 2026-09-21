from __future__ import annotations

from typing import Any

import httpx
import streamlit as st

from client import ClientConfig, ConversationClient


st.set_page_config(
    page_title="Movie Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.18); }
      .movie-agent-kicker { font-size: .82rem; opacity: .68; letter-spacing: .06em; text-transform: uppercase; }
      .movie-agent-subtle { opacity: .68; font-size: .92rem; }
      .movie-agent-chip {
        display: inline-block; padding: .18rem .5rem; margin-right: .35rem; margin-bottom: .25rem;
        border-radius: 999px; border: 1px solid rgba(128,128,128,.28); font-size: .78rem;
      }
      div[data-testid="stStatusWidget"] { border-radius: 14px; }
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


def render_message(message: dict[str, Any]) -> None:
    role = message.get("role")
    if role not in {"user", "assistant"}:
        return
    with st.chat_message(role):
        st.markdown(message.get("content") or "")


def compact_tool_result(payload: dict[str, Any]) -> str:
    tool = payload.get("tool", "tool")
    if payload.get("is_error"):
        return f"{tool} failed"
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


def render_agent_event(status, event_type: str, payload: dict[str, Any], *, show_payloads: bool) -> str | None:
    if event_type == "run_started":
        status.update(label="Understanding your request…", state="running")
        return None

    if event_type == "thinking":
        message = str(payload.get("message") or "Working on the request…")
        status.update(label=message, state="running")
        status.markdown(f"💭 {message}")
        return None

    if event_type == "memory_compaction_started":
        status.markdown("🧠 **Memory:** compacting older conversation context…")
        return None

    if event_type == "memory_compaction_completed":
        status.markdown(
            "🧠 **Memory updated:** "
            f"{payload.get('facts', 0)} facts · {payload.get('constraints', 0)} constraints · "
            f"{payload.get('entities', 0)} entities"
        )
        return None

    if event_type == "memory_compaction_skipped":
        status.warning(str(payload.get("message") or "Memory refresh skipped."))
        return None

    if event_type == "context_recovery":
        status.warning(str(payload.get("message") or "Restarting with a smaller context…"))
        status.update(label="Retrying with focused context…", state="running", expanded=True)
        return None

    if event_type == "plan":
        strategy = str(payload.get("strategy") or "unknown").replace("_", " ")
        summary = payload.get("summary") or "Plan created"
        status.update(label=f"Plan · {strategy}", state="running")
        status.markdown(f"🧭 **Plan:** {summary}")
        criteria = payload.get("extracted_criteria") or []
        if criteria:
            status.markdown("**Extracted criteria**")
            for item in criteria:
                status.markdown(f"- {item}")
        return None

    if event_type == "tool_started":
        tool = payload.get("tool", "tool")
        status.update(label=f"Calling {tool}…", state="running")
        status.markdown(f"🔧 **Calling** `{tool}`")
        if show_payloads and payload.get("arguments") is not None:
            status.json(payload["arguments"])
        return None

    if event_type == "tool_result":
        icon = "❌" if payload.get("is_error") else "✅"
        status.markdown(f"{icon} **Tool result:** {compact_tool_result(payload)}")
        if show_payloads:
            status.json(payload)
        return None

    if event_type == "retrieval_quality":
        quality = str(payload.get("quality") or "unknown")
        icon = {"strong": "🟢", "acceptable": "🟡", "weak": "🟠", "no_match": "🔴"}.get(quality, "⚪")
        status.markdown(f"{icon} {quality_label(payload)}")
        if show_payloads:
            status.json(payload)
        return None

    if event_type == "fallback_available":
        status.markdown(
            "↪️ **Fallback available:** structured evidence was not convincing enough; semantic retrieval may be used."
        )
        return None

    if event_type == "tool_error":
        status.error(
            f"Tool failure: `{payload.get('tool', 'unknown')}` — {payload.get('message', 'unknown error')}"
        )
        if show_payloads:
            status.json(payload)
        return None

    if event_type == "response_started":
        status.update(label="Writing the answer…", state="running")
        return None

    if event_type == "assistant_message":
        return str(payload.get("content") or "")

    if event_type == "completed":
        calls = payload.get("tool_calls", 0)
        failures = payload.get("tool_failures", 0)
        status.update(
            label=f"Completed · {calls} tool call(s) · {failures} failure(s)",
            state="complete",
            expanded=False,
        )
        return None

    if event_type == "error":
        status.error(payload.get("message", "Agent execution failed."))
        status.update(label="Agent run failed", state="error", expanded=True)
        return None

    if show_payloads:
        status.markdown(f"**{event_type}**")
        status.json(payload)
    return None


def sidebar(client: ConversationClient, snapshot: dict[str, Any] | None) -> tuple[bool, bool, bool]:
    with st.sidebar:
        st.markdown('<div class="movie-agent-kicker">Integration Console</div>', unsafe_allow_html=True)
        st.title("🎬 Movie Agent")

        ready, health = client.health()
        if ready:
            st.success("All agent-facing services ready")
        else:
            st.error("Conversation service unavailable")
            with st.expander("Health details"):
                st.json(health)

        if st.button("＋ New conversation", use_container_width=True, disabled=not ready):
            try:
                create_new_conversation(client)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not create conversation: {exc}")

        with st.expander("Load persisted conversation", expanded=False):
            st.text_input("Conversation UUID", key="load_conversation_id", label_visibility="collapsed")
            if st.button("Load conversation", use_container_width=True, disabled=not ready):
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

        if st.session_state.conversation_id:
            st.caption("Conversation")
            st.code(st.session_state.conversation_id, language=None)

        st.divider()
        show_payloads = st.toggle("Show tool payloads", value=False)
        show_debug = st.toggle("Show debug trace", value=False)

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
                st.caption(f"Latest run · {latest.get('status', 'unknown')}")

        if show_debug and st.session_state.last_trace:
            with st.expander("Last SSE trace", expanded=False):
                st.json(st.session_state.last_trace)

    return ready, show_payloads, show_debug


def main() -> None:
    init_state()
    client = get_client()

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

    ready, show_payloads, _ = sidebar(client, snapshot)

    st.markdown('<div class="movie-agent-kicker">Agentic Movie Retrieval</div>', unsafe_allow_html=True)
    st.title("Movie Agent")
    st.markdown(
        '<div class="movie-agent-subtle">Structured/fuzzy catalog search, analytics, semantic retrieval and persisted conversation memory.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    if snapshot:
        for message in snapshot.get("messages", []):
            render_message(message)
    elif ready:
        st.info("Create or load a conversation to begin.")

    prompt = st.chat_input(
        "Ask about a movie, describe a plot, combine filters, or query the catalog…",
        disabled=not ready or not bool(st.session_state.conversation_id),
    )
    if not prompt:
        if snapshot and not snapshot.get("messages"):
            st.caption(
                "Try: “How many movies are there in each genre?” · “Find the movie about entering dreams” · "
                "“Nolan movies after 2010 rated above 8”"
            )
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    answer = ""
    trace: list[dict[str, Any]] = []
    streamed_chars = 0

    with st.chat_message("assistant"):
        status = st.status("Starting agent run…", expanded=True)
        answer_placeholder = st.empty()
        try:
            for event_type, payload in client.stream_message(st.session_state.conversation_id, prompt):
                if event_type != "assistant_delta":
                    trace.append({"event": event_type, "data": payload})

                if event_type == "assistant_delta":
                    delta = str(payload.get("delta") or "")
                    if delta:
                        answer += delta
                        streamed_chars += len(delta)
                        answer_placeholder.markdown(answer + "▌")
                    continue

                candidate_answer = render_agent_event(
                    status,
                    event_type,
                    payload,
                    show_payloads=show_payloads,
                )
                if candidate_answer is not None:
                    answer = candidate_answer
                    answer_placeholder.markdown(answer)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:1000]
            status.error(f"Conversation API returned HTTP {exc.response.status_code}: {detail}")
            status.update(label="Request failed", state="error", expanded=True)
        except Exception as exc:
            status.error(str(exc))
            status.update(label="Request failed", state="error", expanded=True)

        if answer:
            answer_placeholder.markdown(answer)
        else:
            answer_placeholder.warning("No assistant answer was returned. Inspect the execution trace for details.")

    if streamed_chars:
        trace.append({"event": "stream_summary", "data": {"streamed_chars": streamed_chars}})
    st.session_state.last_trace = trace


if __name__ == "__main__":
    main()
