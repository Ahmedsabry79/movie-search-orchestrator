from __future__ import annotations

from typing import Any

import httpx
import streamlit as st

from client import ClientConfig, ConversationClient


st.set_page_config(
    page_title="Movie Agent Test Console",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
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
        title="Streamlit integration test",
        metadata={"client": "streamlit-test-console"},
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
    content = message.get("content") or ""
    with st.chat_message(role):
        st.markdown(content)


def compact_tool_result(payload: dict[str, Any]) -> str:
    tool = payload.get("tool", "tool")
    if payload.get("is_error"):
        return f"{tool} failed"
    if payload.get("count") is not None:
        return f"{tool} returned {payload['count']} result(s)"
    if payload.get("returned_count") is not None:
        return f"{tool} returned {payload['returned_count']} result(s)"
    if payload.get("status"):
        return f"{tool}: {payload['status']}"
    return f"{tool} completed"


def render_agent_event(
    status,
    event_type: str,
    payload: dict[str, Any],
    *,
    show_payloads: bool,
) -> str | None:
    if event_type == "run_started":
        status.update(label="Planning movie search…", state="running")
        return None

    if event_type == "plan":
        strategy = payload.get("strategy", "unknown")
        summary = payload.get("summary", "Plan created")
        status.update(label=f"Plan: {strategy.replace('_', ' ')}", state="running")
        status.markdown(f"**Plan:** {summary}")
        criteria = payload.get("extracted_criteria") or []
        if criteria:
            status.markdown("**Extracted criteria**")
            for item in criteria:
                status.markdown(f"- {item}")
        return None

    if event_type == "tool_started":
        tool = payload.get("tool", "tool")
        status.update(label=f"Running `{tool}`…", state="running")
        status.markdown(f"**Tool call:** `{tool}`")
        if show_payloads and payload.get("arguments") is not None:
            status.json(payload["arguments"])
        return None

    if event_type == "tool_result":
        status.markdown(f"**Result:** {compact_tool_result(payload)}")
        if show_payloads:
            status.json(payload)
        return None

    if event_type == "fallback_available":
        status.warning(
            "Structured retrieval did not produce a strong match. Semantic retrieval is now available as fallback."
        )
        return None

    if event_type == "tool_error":
        status.error(f"Tool failure: `{payload.get('tool', 'unknown')}` — {payload.get('message', 'unknown error')}")
        if show_payloads:
            status.json(payload)
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


def sidebar(client: ConversationClient, snapshot: dict[str, Any] | None) -> tuple[bool, bool]:
    with st.sidebar:
        st.title("Movie Agent")
        st.caption("Integration test console")

        ready, health = client.health()
        if ready:
            st.success("Conversation service ready")
        else:
            st.error("Conversation service unavailable")
            with st.expander("Health details"):
                st.json(health)

        st.divider()

        if st.button("New conversation", use_container_width=True, disabled=not ready):
            try:
                create_new_conversation(client)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not create conversation: {exc}")

        st.text_input(
            "Load persisted conversation",
            key="load_conversation_id",
            placeholder="conversation UUID",
        )
        if st.button("Load", use_container_width=True, disabled=not ready):
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
            st.caption("Current conversation")
            st.code(st.session_state.conversation_id, language=None)

        st.divider()
        show_payloads = st.toggle("Show tool payloads", value=False)
        show_persistence = st.toggle("Show persistence state", value=True)

        if snapshot and show_persistence:
            st.subheader("Persisted state")
            st.metric("Messages", len(snapshot.get("messages", [])))
            st.metric("Runs", len(snapshot.get("runs", [])))
            st.metric("Events", len(snapshot.get("events", [])))
            runs = snapshot.get("runs", [])
            if runs:
                latest = runs[0]
                st.caption(f"Latest run: {latest.get('status', 'unknown')}")

        if st.session_state.last_trace:
            with st.expander("Last raw SSE trace"):
                st.json(st.session_state.last_trace)

    return ready, show_payloads


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

    ready, show_payloads = sidebar(client, snapshot)

    st.title("Movie Agent Test Console")
    st.caption(
        "Tests the full persisted flow: Streamlit → conversation service → Gemma → MCP → DB/Vector services."
    )

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    if snapshot:
        for message in snapshot.get("messages", []):
            render_message(message)
    elif ready:
        st.info("Create or load a conversation to begin.")

    prompt = st.chat_input(
        "Describe the movie(s) you want to find…",
        disabled=not ready or not bool(st.session_state.conversation_id),
    )
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    answer = ""
    trace: list[dict[str, Any]] = []

    with st.chat_message("assistant"):
        with st.status("Starting agent run…", expanded=True) as status:
            try:
                for event_type, payload in client.stream_message(st.session_state.conversation_id, prompt):
                    trace.append({"event": event_type, "data": payload})
                    candidate_answer = render_agent_event(
                        status,
                        event_type,
                        payload,
                        show_payloads=show_payloads,
                    )
                    if candidate_answer is not None:
                        answer = candidate_answer
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text[:1000]
                status.error(f"Conversation API returned HTTP {exc.response.status_code}: {detail}")
                status.update(label="Request failed", state="error", expanded=True)
            except Exception as exc:
                status.error(str(exc))
                status.update(label="Request failed", state="error", expanded=True)

        if answer:
            st.markdown(answer)
        else:
            st.warning("No assistant answer was returned. Inspect the execution trace for details.")

    st.session_state.last_trace = trace


if __name__ == "__main__":
    main()
